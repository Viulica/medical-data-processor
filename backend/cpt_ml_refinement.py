"""ML-based CPT refinement for PCE-PMC and PCE-WWMG colonoscopy cases.

Loads pre-trained Random Forest models and applies them to the unified pipeline's
output dataframe. Only second-guesses Gemini when:
  - worktracker_group is PCE-PMC or PCE-WWMG
  - Gemini predicted 00811 or 00812 (the ambiguous codes)
  - All other predictions (00731, 00813, 00142, etc.) pass through unchanged

For each candidate row, runs the group-specific RF on extracted features and
applies reject-option thresholds calibrated for ≥98% precision on both classes.

Outputs:
  - 'Procedure Code' (potentially overridden)
  - 'ML Refinement' (label describing what happened)
  - 'Needs Review' (True when model is uncertain — flag for human)
"""
import os
import re
import pickle
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Reject-option thresholds — calibrated on 900 PMC + 635 WWMG cross-validated cases
# for ≥98% precision on both 00811 and 00812 simultaneously.
GROUP_THRESHOLDS = {
    "PCE-PMC":  {"low": 0.42, "high": 0.69},
    "PCE-WWMG": {"low": 0.21, "high": 0.64},
}

GROUP_MODEL_FILES = {
    "PCE-PMC":  "pce_pmc_cpt.pkl",
    "PCE-WWMG": "pce_wwmg_cpt.pkl",
}

# Feature schema — must match the order/names used at training time
BOOL_FIELDS = [
    'indication_has_screening_word', 'indication_has_surveillance_word', 'indication_has_z1211',
    'indication_has_z86010', 'indication_has_z800', 'cologuard_or_fit_positive', 'has_ibd_diagnosis',
    'has_personal_hx_colon_cancer', 'has_family_hx_colon_cancer', 'polyp_found_this_visit',
    'symptomatic_indication', 'abnormal_imaging_indication', 'is_medicare_or_ma', 'is_medicaid',
    'first_time_colonoscopy', 'prior_polyps_mentioned', 'dx_has_active_d12', 'gi_billed_polypectomy_or_biopsy',
    'has_high_grade_dysplasia', 'has_advanced_adenoma', 'family_hx_colon_cancer_checkbox',
    'family_hx_colon_polyps_checkbox', 'personal_hx_colon_adenomas_checkbox',
    'anesthesia_preop_cancer_positive', 'lynch_or_hereditary_syndrome',
]

# Models loaded lazily on first use (avoids import-time cost when not needed)
_MODELS = {}


def _models_dir() -> Path:
    return Path(__file__).resolve().parent / "models"


def _load_model(group: str):
    if group in _MODELS:
        return _MODELS[group]
    fname = GROUP_MODEL_FILES.get(group)
    if not fname:
        return None
    path = _models_dir() / fname
    if not path.exists():
        logger.warning(f"CPT ML model not found for {group}: {path}")
        _MODELS[group] = None
        return None
    with open(path, "rb") as f:
        artifact = pickle.load(f)
    _MODELS[group] = artifact
    logger.info(f"Loaded CPT ML model for {group} from {path}")
    return artifact


def _yes(v) -> int:
    return int(str(v).strip().upper() == "YES")


def _build_features(row, feat_cols):
    """Build numeric feature vector from a unified pipeline row.

    Row should have the 30 ML extraction fields populated by templates 155/100.
    """
    gi_codes = str(row.get("gi_procedure_codes", "")).upper()
    proc_label = str(row.get("procedure_label_text", "")).upper().strip()

    derived = {
        "gi_g0121": int("G0121" in gi_codes),
        "gi_multi_cpts": int(len(re.findall(r"\b(?:45[0-9]{3}|G01\d\d)\b", gi_codes)) >= 2),
        "proc_label_screening": int("SCREENING" in proc_label),
        "proc_label_pure": int(bool(re.match(r"^COLONOSCOPY\s*\(", proc_label))),
    }

    vec = []
    for fc in feat_cols:
        if fc.endswith("_b"):
            vec.append(_yes(row.get(fc[:-2], "")))
        elif fc in derived:
            vec.append(derived[fc])
        else:
            # Unknown feature column — default to 0
            vec.append(0)
    return [vec]


def refine_cpt(row, gemini_pred: str, group: str):
    """Refine a single CPT prediction.

    Returns (final_cpt, refinement_label, needs_review_bool).

    Behaviour:
      - Non-PMC/WWMG groups: pass-through, no refinement.
      - Pred not in {00811, 00812}: pass-through (Gemini is ~100% on those).
      - PMC/WWMG + 00811/00812: run RF, apply reject-option thresholds.
    """
    pred = str(gemini_pred or "").strip()

    # Group filter
    if group not in GROUP_MODEL_FILES:
        return pred, "no_model_for_group", False

    # Only refine the ambiguous binary codes
    if pred not in ("00811", "00812"):
        return pred, "passthrough_non_binary", False

    artifact = _load_model(group)
    if artifact is None:
        return pred, "model_unavailable", False

    model = artifact["model"]
    feat_cols = artifact["feat_cols"]
    classes = list(getattr(model, "classes_", [0, 1]))
    pos_idx = classes.index(1) if 1 in classes else 1

    try:
        features = _build_features(row, feat_cols)
        proba = float(model.predict_proba(features)[0, pos_idx])
    except Exception as e:
        logger.exception(f"CPT ML refine failed for group={group}, row keys={list(row.keys())[:5]}: {e}")
        return pred, f"ml_error:{type(e).__name__}", False

    thr = GROUP_THRESHOLDS[group]

    if proba >= thr["high"]:
        return "00811", f"ml_confident_811:p={proba:.2f}", False
    elif proba <= thr["low"]:
        return "00812", f"ml_confident_812:p={proba:.2f}", False
    else:
        # Uncertain zone — keep Gemini's pick but flag for review
        return pred, f"ml_uncertain:p={proba:.2f}", True


def refine_dataframe(df, group: str):
    """Apply CPT refinement to every row of a unified pipeline DataFrame.

    Mutates df in-place by:
      - Updating `Procedure Code` for high-confidence ML overrides
      - Adding `ML Refinement` column with the refinement label
      - Adding `Needs Review` boolean column
      - Adding `ML Original CPT` column preserving Gemini's original prediction

    Returns counts dict for logging.
    """
    print(f"[CPT-ML] refine_dataframe called: group={group}, rows={len(df)}", flush=True)
    logger.info(f"[CPT-ML] refine_dataframe called: group={group}, rows={len(df)}")

    if group not in GROUP_MODEL_FILES:
        msg = f"[CPT-ML] group {group} has no ML model — skipping refinement"
        print(msg, flush=True); logger.info(msg)
        df["ML Refinement"] = "no_model_for_group"
        df["Needs Review"] = False
        df["ML Original CPT"] = df.get("Procedure Code", "")
        return {"total": len(df), "refined": 0, "review": 0, "no_op": len(df)}

    if "Procedure Code" not in df.columns:
        msg = f"[CPT-ML] no 'Procedure Code' column in df, skipping"
        print(msg, flush=True); logger.warning(msg)
        df["ML Refinement"] = "no_cpt_column"
        df["Needs Review"] = False
        return {"total": len(df), "refined": 0, "review": 0, "no_op": len(df)}

    artifact = _load_model(group)
    if artifact is None:
        msg = f"[CPT-ML] model artifact missing for {group}, skipping refinement"
        print(msg, flush=True); logger.warning(msg)
        df["ML Refinement"] = "model_unavailable"
        df["Needs Review"] = False
        return {"total": len(df), "refined": 0, "review": 0, "no_op": len(df)}

    print(f"[CPT-ML] model loaded for {group}: n_train={artifact.get('n_train')}, n_features={len(artifact['feat_cols'])}", flush=True)

    original = df["Procedure Code"].astype(str).str.strip().tolist()
    new_preds = []
    labels = []
    reviews = []
    sample_rows = []   # collect first 5 transitions for diagnostics

    for idx, orig_pred in enumerate(original):
        row = df.iloc[idx].to_dict()
        new_pred, label, needs_review = refine_cpt(row, orig_pred, group)
        new_preds.append(new_pred)
        labels.append(label)
        reviews.append(needs_review)
        if (orig_pred != new_pred or needs_review) and len(sample_rows) < 5:
            sample_rows.append((row.get("source_file", "?"), orig_pred, new_pred, label))

    df["ML Original CPT"] = original
    df["Procedure Code"] = new_preds
    df["ML Refinement"] = labels
    df["Needs Review"] = reviews

    refined = sum(1 for o, n in zip(original, new_preds) if o != n)
    review = sum(1 for r in reviews if r)
    no_op = sum(1 for l in labels if l in ("passthrough_non_binary", "no_model_for_group"))

    summary = (
        f"[CPT-ML] refinement [{group}] DONE: total={len(df)}, "
        f"overridden={refined}, needs_review={review}, passthrough={no_op}"
    )
    print(summary, flush=True); logger.info(summary)

    for src, orig, new, lbl in sample_rows:
        line = f"[CPT-ML]   sample: {src} | {orig} -> {new} | {lbl}"
        print(line, flush=True); logger.info(line)

    return {"total": len(df), "refined": refined, "review": review, "no_op": no_op}

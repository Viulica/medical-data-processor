"""
RIV ensemble auto-posting gate.

Combines the LLM CPT prediction with a procedure-text RandomForest model.
A case is eligible for auto-posting when:
  - The LLM prediction is in WHITELIST (00811/00812/00813/00731), OR
  - The LLM and RF predictions agree AND that code is in HP_CODES.

Otherwise the case is flagged for coder review.

Activated whenever the worktracker group contains "RIV".
"""
from __future__ import annotations

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_MODELS_DIR = Path(__file__).parent / "models"
_MODEL_PATH = _MODELS_DIR / "riv_rf_model.pkl"
_HP_PATH = _MODELS_DIR / "riv_rf_hp.json"

_cached: Dict[str, object] = {}


def _load_model():
    if "model" in _cached:
        return _cached["model"]
    if not _MODEL_PATH.exists():
        raise FileNotFoundError(f"RIV RF model not found at {_MODEL_PATH}")
    with open(_MODEL_PATH, "rb") as f:
        m = pickle.load(f)
    _cached["model"] = m
    return m


def _load_hp() -> Tuple[set, set]:
    if "hp" in _cached:
        return _cached["hp"]
    if not _HP_PATH.exists():
        raise FileNotFoundError(f"RIV HP config not found at {_HP_PATH}")
    with open(_HP_PATH) as f:
        d = json.load(f)
    out = (set(d["hp_codes"]), set(d["whitelist"]))
    _cached["hp"] = out
    return out


def _norm_code(c) -> str:
    """Normalize a CPT code to 5-digit zero-padded string. Returns '' if blank."""
    if c is None:
        return ""
    s = str(c).strip()
    if not s or s.lower() in ("nan", "none", "null"):
        return ""
    # strip trailing .0 from pandas-loaded ints
    if s.endswith(".0"):
        s = s[:-2]
    if s.isdigit() and len(s) == 4:
        s = "0" + s
    return s


def is_riv_group(worktracker_group: Optional[str]) -> bool:
    """RIV gate activates for any group whose name contains 'RIV' (case-insensitive)."""
    if not worktracker_group:
        return False
    return "RIV" in str(worktracker_group).upper()


def predict_rf(proc_texts: List[str]) -> List[str]:
    """Predict CPT for each procedure_text via the RF model."""
    if not proc_texts:
        return []
    m = _load_model()
    vec = m["vec"]
    clf = m["clf"]
    # Normalize inputs
    cleaned = [(t or "").strip() for t in proc_texts]
    X = vec.transform(cleaned)
    preds = clf.predict(X)
    return [str(p) for p in preds]


def apply_riv_gate(df, cpt_col: str = "Procedure Code",
                   proc_text_col: str = "procedure_text") -> Dict[str, int]:
    """
    Apply the RIV ensemble gate IN PLACE to df.

    Adds two columns: 'CPT Disposition' and writes 'CPT' into 'CoderVerify'
    (creating the column if missing) for any non-auto-post row.

    Returns a stats dict.
    """
    hp_codes, whitelist = _load_hp()

    # Resolve columns
    if cpt_col not in df.columns:
        alt = "ASA Code" if "ASA Code" in df.columns else None
        if alt is None:
            logger.warning("[RIV gate] No CPT column found; skipping")
            return {"total": len(df), "auto_post": 0, "review": 0, "skipped": True}
        cpt_col = alt

    # Procedure text column — required. Tolerate missing-but-warn.
    have_proc = proc_text_col in df.columns
    if not have_proc:
        logger.warning(
            f"[RIV gate] '{proc_text_col}' column not present — every non-whitelist "
            "row will fall through to coder review (no RF vote possible)."
        )

    llm_preds = [_norm_code(c) for c in df[cpt_col].tolist()]

    if have_proc:
        proc_texts = [str(t) if t is not None else "" for t in df[proc_text_col].tolist()]
        rf_preds_raw = predict_rf(proc_texts)
        rf_preds = [_norm_code(p) for p in rf_preds_raw]
    else:
        rf_preds = [""] * len(df)

    dispositions: List[str] = []
    coder_flag: List[str] = []
    n_wl = n_agree = n_review = 0

    for i, llm in enumerate(llm_preds):
        rf = rf_preds[i]
        proc = (df[proc_text_col].iloc[i] if have_proc else "")
        proc_blank = (not proc) or str(proc).strip() == "" or str(proc).strip().upper() == "UNCLEAR"
        if not llm:
            dispositions.append("NO_PREDICTION"); coder_flag.append("CPT"); n_review += 1
            continue
        if llm in whitelist:
            dispositions.append("AUTO_POST_WHITELIST"); coder_flag.append(""); n_wl += 1
            continue
        if not have_proc or proc_blank:
            # Can't vote — coder review
            dispositions.append("REVIEW_NO_PROC_TEXT"); coder_flag.append("CPT"); n_review += 1
            continue
        if rf == llm and llm in hp_codes:
            dispositions.append("AUTO_POST_AGREE_HP"); coder_flag.append(""); n_agree += 1
            continue
        if rf == llm and llm not in hp_codes:
            dispositions.append("REVIEW_AGREE_NOT_HP"); coder_flag.append("CPT"); n_review += 1
            continue
        # Disagreement
        dispositions.append("REVIEW_DISAGREE"); coder_flag.append("CPT"); n_review += 1

    # Write columns
    df["CPT Disposition"] = dispositions
    df["RF Prediction"] = rf_preds
    # Merge CoderVerify
    if "CoderVerify" in df.columns:
        merged = []
        for i, flag in enumerate(coder_flag):
            existing = str(df["CoderVerify"].iloc[i] or "").strip()
            if existing.lower() in ("nan", "none", "null"):
                existing = ""
            if flag and existing and flag not in existing.split(", "):
                merged.append(f"{flag}, {existing}")
            elif flag and not existing:
                merged.append(flag)
            else:
                merged.append(existing)
        df["CoderVerify"] = merged
    else:
        df["CoderVerify"] = coder_flag

    stats = {
        "total": len(df),
        "auto_post_whitelist": n_wl,
        "auto_post_agree_hp": n_agree,
        "review": n_review,
        "skipped": False,
    }
    stats["auto_post_total"] = n_wl + n_agree
    stats["auto_post_pct"] = round(100.0 * (n_wl + n_agree) / max(1, len(df)), 1)
    return stats

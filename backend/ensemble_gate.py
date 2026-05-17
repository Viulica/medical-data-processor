"""
Generic LLM + RandomForest CPT auto-post gate.

For each registered group, this module loads a per-group RF model trained on
procedure_text → coder-corrected CPT pairs and decides per row:

  - If the LLM CPT prediction is in the group's WHITELIST  → auto-post.
  - Else if LLM == RF prediction AND that code is in the group's HP set → auto-post.
  - Otherwise → flag for coder review.

Groups are matched by case-insensitive substring on worktracker_group.

----------------------------------------------------------------------------
Per-group artifacts live under backend/models/:
  - <prefix>_rf_model.pkl   : pickled {"vec": TfidfVectorizer, "clf": RandomForest}
  - <prefix>_rf_hp.json     : {"hp_codes": [...], "whitelist": [...]}

To add a new group, drop the two files above and add an entry to GROUP_CONFIGS.
----------------------------------------------------------------------------

NOTE — STAFF/CODER VERIFY POPULATION IS CURRENTLY DISABLED.
This module computes the gate disposition and the would-be CoderVerify tag, but
the production write of StaffVerify/CoderVerify to the output DataFrame is
commented out below (search for "WRITE_VERIFY_COLUMNS"). Re-enable by flipping
WRITE_VERIFY_COLUMNS to True (or per-group via PER_GROUP_WRITE_VERIFY) once the
downstream consumers are ready.
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_MODELS_DIR = Path(__file__).parent / "models"

# === REGISTRY ============================================================
# Each entry defines a group: how to match it, which model files to use,
# and (optionally) per-group flags. The matcher is case-insensitive.
GROUP_CONFIGS: Dict[str, Dict] = {
    "RIV": {
        "matcher":    lambda g: "RIV" in g.upper(),
        "model_path": "riv_rf_model.pkl",
        "hp_path":    "riv_rf_hp.json",
    },
    "DUN": {
        "matcher":    lambda g: "DUN" in g.upper(),
        "model_path": "dun_rf_model.pkl",
        "hp_path":    "dun_rf_hp.json",
    },
}

# === FEATURE FLAGS =======================================================
# Master switch: when False, the gate computes the disposition but does NOT
# write CoderVerify / StaffVerify to the DataFrame. CPT Disposition and
# RF Prediction columns are always written (they're diagnostic only).
WRITE_VERIFY_COLUMNS: bool = False

# Per-group override (None = use master switch). Useful when you want to
# enable the gate for one group but not yet another.
PER_GROUP_WRITE_VERIFY: Dict[str, bool] = {
    # "RIV": True,  # uncomment to enable RIV CoderVerify writes
    # "DUN": True,  # uncomment to enable DUN CoderVerify writes
}

_cache: Dict[str, Dict] = {}


def find_group_config(worktracker_group: Optional[str]) -> Tuple[Optional[str], Optional[Dict]]:
    """Return (group_name, config) for the first matching registry entry, or (None, None)."""
    if not worktracker_group:
        return None, None
    for name, cfg in GROUP_CONFIGS.items():
        try:
            if cfg["matcher"](worktracker_group):
                return name, cfg
        except Exception:
            continue
    return None, None


def is_gated_group(worktracker_group: Optional[str]) -> bool:
    return find_group_config(worktracker_group)[1] is not None


def _load_group(name: str, cfg: Dict) -> Dict:
    """Load and cache the RF model + HP json for a group."""
    if name in _cache:
        return _cache[name]
    model_path = _MODELS_DIR / cfg["model_path"]
    hp_path    = _MODELS_DIR / cfg["hp_path"]
    if not model_path.exists():
        raise FileNotFoundError(f"{name} RF model not found at {model_path}")
    if not hp_path.exists():
        raise FileNotFoundError(f"{name} HP config not found at {hp_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(hp_path) as f:
        hp = json.load(f)
    _cache[name] = {
        "vec":       model["vec"],
        "clf":       model["clf"],
        "hp_codes":  set(hp["hp_codes"]),
        "whitelist": set(hp["whitelist"]),
    }
    return _cache[name]


def _norm_code(c) -> str:
    """Normalize a CPT code to 5-digit zero-padded string. Returns '' if blank."""
    if c is None:
        return ""
    s = str(c).strip()
    if not s or s.lower() in ("nan", "none", "null"):
        return ""
    if s.endswith(".0"):
        s = s[:-2]
    if s.isdigit() and len(s) == 4:
        s = "0" + s
    return s


def predict_rf(group_name: str, proc_texts: List[str]) -> List[str]:
    cfg = GROUP_CONFIGS.get(group_name)
    if cfg is None:
        return [""] * len(proc_texts)
    loaded = _load_group(group_name, cfg)
    cleaned = [(t or "").strip() for t in proc_texts]
    X = loaded["vec"].transform(cleaned)
    preds = loaded["clf"].predict(X)
    return [str(p) for p in preds]


def apply_gate(df, worktracker_group: str,
               cpt_col: str = "Procedure Code",
               proc_text_col: str = "procedure_text") -> Optional[Dict]:
    """
    Apply the ensemble gate IN PLACE.

    Always writes diagnostic columns:
      - 'CPT Disposition' : AUTO_POST_WHITELIST | AUTO_POST_AGREE_HP |
                            REVIEW_DISAGREE | REVIEW_AGREE_NOT_HP |
                            REVIEW_NO_PROC_TEXT | NO_PREDICTION
      - 'RF Prediction'   : the RF's CPT vote per row

    Writes 'CoderVerify' only when WRITE_VERIFY_COLUMNS (or the per-group
    override) is True. The 'CPT' rule tag is otherwise computed but not
    persisted.

    Returns a stats dict, or None if the group has no registered config.
    """
    name, cfg = find_group_config(worktracker_group)
    if cfg is None:
        return None

    loaded = _load_group(name, cfg)
    hp_codes = loaded["hp_codes"]
    whitelist = loaded["whitelist"]

    # Resolve CPT column
    if cpt_col not in df.columns:
        alt = "ASA Code" if "ASA Code" in df.columns else None
        if alt is None:
            logger.warning(f"[{name} gate] No CPT column found; skipping")
            return {"group": name, "total": len(df), "skipped": True,
                    "reason": "no_cpt_col"}
        cpt_col = alt

    have_proc = proc_text_col in df.columns
    if not have_proc:
        logger.warning(
            f"[{name} gate] '{proc_text_col}' column not present — every "
            "non-whitelist row will fall through to coder review."
        )

    llm_preds = [_norm_code(c) for c in df[cpt_col].tolist()]
    if have_proc:
        proc_texts = [str(t) if t is not None else "" for t in df[proc_text_col].tolist()]
        rf_preds = [_norm_code(p) for p in predict_rf(name, proc_texts)]
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
            dispositions.append("REVIEW_NO_PROC_TEXT"); coder_flag.append("CPT"); n_review += 1
            continue
        if rf == llm and llm in hp_codes:
            dispositions.append("AUTO_POST_AGREE_HP"); coder_flag.append(""); n_agree += 1
            continue
        if rf == llm and llm not in hp_codes:
            dispositions.append("REVIEW_AGREE_NOT_HP"); coder_flag.append("CPT"); n_review += 1
            continue
        dispositions.append("REVIEW_DISAGREE"); coder_flag.append("CPT"); n_review += 1

    # Diagnostic columns — always written
    df["CPT Disposition"] = dispositions
    df["RF Prediction"] = rf_preds

    # ----------------------------------------------------------------
    # WRITE_VERIFY_COLUMNS — currently DISABLED.
    # When ready to populate StaffVerify/CoderVerify based on the gate,
    # set WRITE_VERIFY_COLUMNS=True at the top of this file, or add an
    # entry to PER_GROUP_WRITE_VERIFY (e.g. {"RIV": True}).
    # ----------------------------------------------------------------
    write_verify = PER_GROUP_WRITE_VERIFY.get(name, WRITE_VERIFY_COLUMNS)
    if write_verify:
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
        # StaffVerify pass-through (no rule tag prepended here)
        if "StaffVerify" not in df.columns:
            df["StaffVerify"] = ""

    stats = {
        "group": name,
        "total": len(df),
        "auto_post_whitelist": n_wl,
        "auto_post_agree_hp": n_agree,
        "review": n_review,
        "auto_post_total": n_wl + n_agree,
        "auto_post_pct": round(100.0 * (n_wl + n_agree) / max(1, len(df)), 1),
        "verify_written": write_verify,
        "skipped": False,
    }
    return stats


# ---- Backward-compat shims (kept so existing imports keep working) -------
def is_riv_group(g):  # pragma: no cover
    return find_group_config(g)[0] == "RIV"

def apply_riv_gate(df, cpt_col="Procedure Code", proc_text_col="procedure_text"):
    # Delegates to apply_gate for any RIV group
    return apply_gate(df, "RIV-COMPAT", cpt_col=cpt_col, proc_text_col=proc_text_col)

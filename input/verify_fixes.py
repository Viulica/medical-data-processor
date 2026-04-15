"""Re-run predictions on the 9 error PDFs using updated instruction templates.
Compare to ground truth from the cached allCharges output.
"""
import os, sys, shutil, tempfile, csv
sys.path.insert(0, "backend")
sys.path.insert(0, "backend/general-coding")

# Load env
from dotenv import load_dotenv
load_dotenv("backend/.env")

from db_utils import get_prediction_instruction
from predict_general import predict_codes_from_pdfs_api, predict_icd_codes_from_pdfs_api

MODEL = "google/gemini-3-flash-preview"
N_PAGES = 6  # cover full anesthesia record
MAX_WORKERS = 10

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ERROR_DIR = os.path.join(HERE, "error_pdfs")
OUT_DIR = os.path.join(HERE, "verify_output")
os.makedirs(OUT_DIR, exist_ok=True)

# Ground truth (group -> patient_last -> (cpt, icd1))
GT = {
    "LOV_162": {
        "MORRIS":    ("00142", "H25.12"),
        "DEAN":      ("00142", "H25.12"),
        "FREUND":    ("00142", "H25.12"),
    },
    "INJE-CLIK_119": {
        "KIESOW":    ("00140", "H40.9"),
        "SWARTZ":    ("00140", "H40.9"),
        "SNYDER":    ("00140", "H40.9"),
    },
    "WPA_339": {
        "DAVIS":     ("00140", "H40.1111"),
        "HEDGCOTH":  ("00140", "H40.1121"),
        "SHEPARD":   ("00140", "H40.1121"),
    },
}

# Template IDs
LOV_CPT_TID  = 139  # LOV-ASC CPT
INJE_ICD_TID = 126  # INJE-CLIK ICD
WPA_ICD_TID  = 58   # WPA ICD

cpt_tmpl = {
    "LOV_162":       get_prediction_instruction(instruction_id=LOV_CPT_TID)["instructions_text"],
}
icd_tmpl = {
    "INJE-CLIK_119": get_prediction_instruction(instruction_id=INJE_ICD_TID)["instructions_text"],
    "WPA_339":       get_prediction_instruction(instruction_id=WPA_ICD_TID)["instructions_text"],
}

def run_cpt(bucket):
    folder = os.path.join(ERROR_DIR, bucket)
    out = os.path.join(OUT_DIR, f"{bucket}_cpt.csv")
    ok = predict_codes_from_pdfs_api(
        pdf_folder=folder, output_file=out, n_pages=N_PAGES,
        model=MODEL, custom_instructions=cpt_tmpl[bucket],
        max_workers=MAX_WORKERS,
    )
    return out if ok else None

def run_icd(bucket):
    folder = os.path.join(ERROR_DIR, bucket)
    out = os.path.join(OUT_DIR, f"{bucket}_icd.csv")
    ok = predict_icd_codes_from_pdfs_api(
        pdf_folder=folder, output_file=out, n_pages=N_PAGES,
        model=MODEL, custom_instructions=icd_tmpl[bucket],
        max_workers=MAX_WORKERS,
    )
    return out if ok else None

def parse_csv(path):
    rows = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            fname = r.get("filename") or r.get("PDF") or r.get("File") or ""
            key = os.path.splitext(os.path.basename(fname))[0].upper()
            rows[key] = r
    return rows

print("=" * 70)
print("Running CPT predictions for LOV...")
lov_csv = run_cpt("LOV_162")
print(f"  -> {lov_csv}")

print("\nRunning ICD predictions for INJE-CLIK...")
inje_csv = run_icd("INJE-CLIK_119")
print(f"  -> {inje_csv}")

print("\nRunning ICD predictions for WPA...")
wpa_csv = run_icd("WPA_339")
print(f"  -> {wpa_csv}")

# Compare
print("\n" + "=" * 70)
print("VERIFICATION RESULTS")
print("=" * 70)

total, passed = 0, 0

if lov_csv and os.path.exists(lov_csv):
    rows = parse_csv(lov_csv)
    print(f"\n--- LOV (CPT test) ---")
    print(f"{'Patient':<10}{'AI CPT':>10}{'GT CPT':>10}  Status")
    for patient, (gt_cpt, _) in GT["LOV_162"].items():
        total += 1
        # Match by filename containing patient name
        match = None
        for k, v in rows.items():
            if patient in k:
                match = v; break
        ai_cpt = (match.get("cpt_code") or match.get("CPT") or match.get("predicted_cpt") or "?") if match else "?"
        ok = ai_cpt == gt_cpt
        if ok: passed += 1
        print(f"{patient:<10}{ai_cpt:>10}{gt_cpt:>10}  {'PASS' if ok else 'FAIL'}")

if inje_csv and os.path.exists(inje_csv):
    rows = parse_csv(inje_csv)
    print(f"\n--- INJE-CLIK (ICD1 test) ---")
    print(f"{'Patient':<10}{'AI ICD1':>12}{'GT ICD1':>12}  Status")
    for patient, (_, gt_icd) in GT["INJE-CLIK_119"].items():
        total += 1
        match = None
        for k, v in rows.items():
            if patient in k:
                match = v; break
        ai_icd = (match.get("icd1") or match.get("ICD1") or match.get("predicted_icd1") or "?") if match else "?"
        ok = ai_icd == gt_icd
        if ok: passed += 1
        print(f"{patient:<10}{ai_icd:>12}{gt_icd:>12}  {'PASS' if ok else 'FAIL'}")

if wpa_csv and os.path.exists(wpa_csv):
    rows = parse_csv(wpa_csv)
    print(f"\n--- WPA (ICD1 test) ---")
    print(f"{'Patient':<10}{'AI ICD1':>12}{'GT ICD1':>12}  Status")
    for patient, (_, gt_icd) in GT["WPA_339"].items():
        total += 1
        match = None
        for k, v in rows.items():
            if patient in k:
                match = v; break
        ai_icd = (match.get("icd1") or match.get("ICD1") or match.get("predicted_icd1") or "?") if match else "?"
        ok = ai_icd == gt_icd
        if ok: passed += 1
        print(f"{patient:<10}{ai_icd:>12}{gt_icd:>12}  {'PASS' if ok else 'FAIL'}")

print(f"\n{'='*70}\nTotal: {passed}/{total} passed ({passed/total*100:.0f}%)")

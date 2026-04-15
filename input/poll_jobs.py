"""Poll the already-submitted jobs and download results."""
import os, json, ssl, time, urllib.request, openpyxl

BACKEND = "https://medical-data-processor-production.up.railway.app"
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "verify_output")
os.makedirs(OUT_DIR, exist_ok=True)

JOBS = {
    "LOV_162":       "439c3f5c-d928-4166-9876-f2d96b23a65b",
    "INJE-CLIK_119": "4eae4584-ca8b-4c1b-b35f-45dc768329c5",
    "WPA_339":       "0dec1f72-89f1-47aa-8eed-738efe07d57a",
}

GT = {
    "LOV_162": {"MORRIS":("00142","H25.12"),"DEAN":("00142","H25.12"),"FREUND":("00142","H25.12")},
    "INJE-CLIK_119": {"KIESOW":("00140","H40.9"),"SWARTZ":("00140","H40.9"),"SNYDER":("00140","H40.9")},
    "WPA_339": {"DAVIS":("00140","H40.1111"),"HEDGCOTH":("00140","H40.1121"),"SHEPARD":("00140","H40.1121")},
}

# which field to compare per bucket
TEST = {
    "LOV_162": "CPT",
    "INJE-CLIK_119": "ICD1",
    "WPA_339": "ICD1",
}


def get_status(jid):
    req = urllib.request.Request(f"{BACKEND}/status/{jid}")
    with urllib.request.urlopen(req, context=CTX, timeout=30) as r:
        return json.loads(r.read())


def download(jid, path):
    req = urllib.request.Request(f"{BACKEND}/api/unified-results/{jid}/download")
    with urllib.request.urlopen(req, context=CTX, timeout=120) as r:
        data = r.read()
    with open(path, "wb") as fh:
        fh.write(data)


# Poll until all done
pending = set(JOBS)
start = time.time()
while pending:
    for bucket in list(pending):
        jid = JOBS[bucket]
        try:
            st = get_status(jid)
        except Exception as e:
            print(f"  [{bucket}] status error: {e}")
            continue
        status = st.get("status")
        progress = st.get("progress", 0)
        msg = st.get("message", "")
        elapsed = int(time.time() - start)
        print(f"  t={elapsed}s [{bucket}] {status} {progress}% — {msg[:60]}")
        if status in ("completed", "failed"):
            if status == "completed":
                out = os.path.join(OUT_DIR, f"{bucket}.xlsx")
                try:
                    download(jid, out)
                    print(f"    downloaded -> {out}")
                except Exception as e:
                    print(f"    download error: {e}")
            pending.discard(bucket)
    if pending:
        time.sleep(8)

# Load results and compare
print("\n" + "="*70)
print("VERIFICATION vs GROUND TRUTH")
print("="*70)

def load(path):
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb[wb.sheetnames[0]]
    rows = list(ws.iter_rows(values_only=True))
    if not rows: return [], []
    header = [str(c).strip() if c else "" for c in rows[0]]
    return header, [dict(zip(header, r)) for r in rows[1:]]


def find_col(header, candidates):
    for c in candidates:
        for h in header:
            if h and c.lower() == h.lower():
                return h
    for c in candidates:
        for h in header:
            if h and c.lower() in h.lower():
                return h
    return None


total = passed = 0
for bucket in JOBS:
    path = os.path.join(OUT_DIR, f"{bucket}.xlsx")
    if not os.path.exists(path):
        print(f"\n[{bucket}] no result file"); continue
    header, rows = load(path)
    print(f"\n--- {bucket} ({len(rows)} rows) header: {header[:8]}...")
    c_file = find_col(header, ["filename", "file", "pdf", "patient_last_name", "Patient Last Name"])
    c_cpt  = find_col(header, ["cpt_code", "cpt", "predicted_cpt"])
    c_icd  = find_col(header, ["icd1", "icd_1", "predicted_icd1"])
    print(f"    file col={c_file!r}  cpt={c_cpt!r}  icd1={c_icd!r}")
    for patient, (gt_cpt, gt_icd) in GT[bucket].items():
        match = None
        for r in rows:
            row_text = " ".join(str(v or "") for v in r.values()).upper()
            if patient in row_text:
                match = r; break
        if not match:
            print(f"    {patient}: NOT FOUND"); continue
        ai_cpt = str(match.get(c_cpt, "") or "").strip() if c_cpt else ""
        ai_icd = str(match.get(c_icd, "") or "").strip() if c_icd else ""
        target = TEST[bucket]
        if target == "CPT":
            total += 1
            ok = ai_cpt == gt_cpt; passed += int(ok)
            print(f"    {patient:<10} AI_CPT={ai_cpt:<8} GT={gt_cpt:<8} {'PASS' if ok else 'FAIL'}")
        else:
            total += 1
            ok = ai_icd == gt_icd; passed += int(ok)
            print(f"    {patient:<10} AI_ICD1={ai_icd:<10} GT={gt_icd:<10} {'PASS' if ok else 'FAIL'}")

print(f"\n{'='*70}\n  TOTAL: {passed}/{total} passed ({passed/total*100:.0f}%)" if total else "No results")

"""Run extraction + CPT + ICD on aip-prhc zip via local backend."""
import json, os, ssl, time, urllib.request, uuid

BACKEND = "http://localhost:8001"
ZIP = os.path.expanduser("~/Downloads/aip prhc 3 split batches.zip")
OUT = "aip-prhc_result.xlsx"

CTX = ssl.create_default_context(); CTX.check_hostname = False; CTX.verify_mode = ssl.CERT_NONE

with open(ZIP, "rb") as fh:
    zbytes = fh.read()
print(f"Zip: {len(zbytes):,} bytes")

MODEL = "google/gemini-3-flash-preview"

CPT_INSTRUCTIONS = """ADDITIONAL CODING RULES FOR THIS FACILITY:

5. HERNIA REPAIR CODING:
   - Ventral/incisional hernia repair (ICD K43.x) in upper abdomen -> use 00752 (not 00700)
   - Inguinal hernia repair -> use 00830
   - Umbilical hernia repair -> use 00752

6. OB ANESTHESIA CODING:
   - Labor epidural / neuraxial for labor and vaginal delivery -> use 01967
   - Planned cesarean section -> use 01968
   - D&C / missed abortion / incomplete abortion -> use 01965
   - If the case is an OB epidural for labor, do NOT use peripheral block codes (64XXX). The main case code is 01967.
   - 01961 is for surgery of the uterus/cervix (e.g. cerclage), NOT for labor epidurals

7. GENITOURINARY PROCEDURE CODING:
   - Hydrocelectomy / spermatocelectomy (N43.x) -> use 00920 (not 00921)
   - 00921 is specifically for vasectomy only
   - Prostate biopsy (R97.20, elevated PSA) -> use 00902 (not 00920 or 00910)
   - Cystoscopy / cystourethroscopy (N32.x, N35.x) -> use 00910
   - Lithotripsy / ureteroscopy for kidney stones (N20.x) -> use 00918

8. BREAST SURGERY CODING:
   - Breast biopsy / lumpectomy / mastectomy (D05.x, C50.x, N60.x) -> use 01610 (not 00400)
   - 00400 is for integumentary/skin procedures, NOT breast surgery

9. SPINE PROCEDURE CODING:
   - Lumbar laminectomy / discectomy / fusion -> use 01940
   - Cervical spine procedures -> use 00600
   - Percutaneous lumbar injections (facet, medial branch block, epidural steroid injection) -> use 01937 or 01938
   - 01939 is for percutaneous procedures on the pelvis, NOT spine injections
"""

fields = {
    "template_id": "188",
    "enable_extraction": "true",
    "extraction_n_pages": "10",
    "extraction_model": "gemini-3-flash-preview",
    "extraction_max_workers": "50",
    "enable_cpt": "true",
    "cpt_vision_mode": "true",
    "cpt_vision_pages": "10",
    "cpt_vision_model": MODEL,
    "cpt_max_workers": "50",
    "cpt_custom_instructions": CPT_INSTRUCTIONS,
    "enable_icd": "true",
    "icd_n_pages": "10",
    "icd_vision_model": MODEL,
    "icd_max_workers": "50",
}

boundary = f"----b{uuid.uuid4().hex}"
parts = []
for k, v in fields.items():
    parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
parts.append(
    f"--{boundary}\r\nContent-Disposition: form-data; name=\"zip_file\"; "
    f"filename=\"aip-prhc.zip\"\r\nContent-Type: application/zip\r\n\r\n".encode()
    + zbytes + b"\r\n"
)
parts.append(f"--{boundary}--\r\n".encode())
body = b"".join(parts)

req = urllib.request.Request(
    f"{BACKEND}/process-unified", data=body, method="POST",
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
)
print("Submitting...")
with urllib.request.urlopen(req, context=CTX, timeout=300) as r:
    resp = json.loads(r.read())
jid = resp["job_id"]
print(f"job_id: {jid}")

start = time.time()
while True:
    req = urllib.request.Request(f"{BACKEND}/status/{jid}")
    with urllib.request.urlopen(req, context=CTX, timeout=30) as r:
        st = json.loads(r.read())
    s = st.get("status"); p = st.get("progress", 0); m = st.get("message", "")
    print(f"  t={int(time.time()-start)}s {s} {p}% — {m[:75]}")
    if s in ("completed", "failed"): break
    time.sleep(10)

if s != "completed":
    print("FAILED:", st); raise SystemExit(1)

req = urllib.request.Request(f"{BACKEND}/download/{jid}?format=xlsx")
with urllib.request.urlopen(req, context=CTX, timeout=120) as r:
    with open(OUT, "wb") as fh: fh.write(r.read())
print(f"\nDownloaded -> {OUT} ({os.path.getsize(OUT):,} bytes)")

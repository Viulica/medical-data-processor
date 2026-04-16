"""Run extraction + CPT + ICD on eap-sca zip via local backend."""
import json, os, ssl, time, urllib.request, uuid

BACKEND = "http://localhost:8001"
ZIP = os.path.expanduser("~/Downloads/eap sca.zip")
OUT = "eap-sca_result.xlsx"

CTX = ssl.create_default_context(); CTX.check_hostname = False; CTX.verify_mode = ssl.CERT_NONE

with open(ZIP, "rb") as fh:
    zbytes = fh.read()
print(f"Zip: {len(zbytes):,} bytes")

MODEL = "google/gemini-3-flash-preview"

fields = {
    "template_id": "193",
    "enable_extraction": "true",
    "extraction_n_pages": "10",
    "extraction_model": "gemini-3-flash-preview",
    "extraction_max_workers": "50",
    "enable_cpt": "true",
    "cpt_vision_mode": "true",
    "cpt_vision_pages": "10",
    "cpt_vision_model": MODEL,
    "cpt_max_workers": "50",
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
    f"filename=\"eap-sca.zip\"\r\nContent-Type: application/zip\r\n\r\n".encode()
    + zbytes + b"\r\n"
)
parts.append(f"--{boundary}--\r\n".encode())
body = b"".join(parts)

req = urllib.request.Request(
    f"{BACKEND}/process-unified", data=body, method="POST",
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
)
print("Submitting...")
with urllib.request.urlopen(req, context=CTX, timeout=600) as r:
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

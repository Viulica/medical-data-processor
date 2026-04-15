"""Download source PDFs from SharePoint via Graph API for the 12 error cases."""
import os, json, ssl, urllib.parse, urllib.request

TENANT = "0138a897-ff88-4dc2-933b-fad359609873"
CLIENT_ID = "a54eaddf-654d-4f0e-9071-2b8c8ad26942"
CLIENT_SECRET = "MlO8Q~6ARQrwWgbvb6v9qWOuHCWIU3m6MaKsTczK"
DRIVE_ID = "b!ngvjU5cJZkK_4oF06KsHXBNacfJKjWNPmBwGVZ1Z4EPXuNIeAqOqSoPA_SSOtvFY"

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "error_pdfs")
os.makedirs(OUT, exist_ok=True)

# (bucket_dir, source_url)
TARGETS = [
    ("LOV_162",       "IAS/2026/BHS"),
    ("INJE-CLIK_119", "INJE/2026/CLIK"),
    ("WPA_339",       "WPA/2026"),
    ("IAS-FVO_418",   "IAS/2026/IAS-FVO"),
]
EHR_PATHS = {
    "LOV_162/MORRIS": "LOV/2026/ATL/LOV-04-07-2026 #162 SORT/EDWARD_MORRIS_J.PDF",
    "LOV_162/DEAN": "LOV/2026/ATL/LOV-04-07-2026 #162 SORT/PAMELA_DEAN_S.PDF",
    "LOV_162/FREUND": "LOV/2026/ATL/LOV-04-07-2026 #162 SORT/KELLY_FREUND_H.PDF",
    "INJE-CLIK_119/KIESOW": "INJE/2026/CLIK/INJE-CLIK-04-01-26 #119/RHODA_KIESOW.PDF",
    "INJE-CLIK_119/SWARTZ": "INJE/2026/CLIK/INJE-CLIK-04-01-26 #119/DAVID_SWARTZ_B.PDF",
    "INJE-CLIK_119/SNYDER": "INJE/2026/CLIK/INJE-CLIK-04-01-26 #119/DARREL_SNYDER_L.PDF",
    "WPA_339/DAVIS": "WPA/2026/WPA 04-07-2026#339/JERRY_DAVIS_W.PDF",
    "WPA_339/HEDGCOTH": "WPA/2026/WPA 04-07-2026#339/MICHAEL_HEDGCOTH.PDF",
    "WPA_339/SHEPARD": "WPA/2026/WPA 04-07-2026#339/CECELIA_SHEPARD_M.PDF",
    "IAS-FVO_418/WICKMAN": "IAS/2026/IAS-FVO/IAS-FVO-04-08-2026 #418/CHARLES_WICKMAN.PDF",
    "IAS-FVO_418/PATEL": "IAS/2026/IAS-FVO/IAS-FVO-04-08-2026 #418/NISHA_PATEL.PDF",
    "IAS-FVO_418/GEAMAN": "IAS/2026/IAS-FVO/IAS-FVO-04-08-2026 #418/PATRICIA_GEAMAN_A.PDF",
    "CHA-HDH_472/LENNARD": "CHA/2026/CHA-HDH/CHA - HDH - 04-06-2026 # 472/MATTHEW_LENNARD_R.PDF",
    "CHA-HDH_472/BRADLEY": "CHA/2026/CHA-HDH/CHA - HDH - 04-06-2026 # 472/TAYLOR_BRADLEY_A.PDF",
    "CHA-HDH_472/RAKER": "CHA/2026/CHA-HDH/CHA - HDH - 04-06-2026 # 472/DEBORAH_RAKER.PDF",
    "PCE-WWMG_333/TANG": "PCE/2026/PCE-WWMG/PCE-WWMG-04-03-2026 #333/LAICHU_TANG.PDF",
    "PCE-WWMG_333/LUKER": "PCE/2026/PCE-WWMG/PCE-WWMG-04-03-2026 #333/JASON_LUKER.PDF",
    "PCE-WWMG_333/CAMARA": "PCE/2026/PCE-WWMG/PCE-WWMG-04-03-2026 #333/KATHRYN_CAMARA_L.PDF",
}

# Get token
data = urllib.parse.urlencode({
    "client_id": CLIENT_ID, "client_secret": CLIENT_SECRET,
    "scope": "https://graph.microsoft.com/.default", "grant_type": "client_credentials",
}).encode()
req = urllib.request.Request(
    f"https://login.microsoftonline.com/{TENANT}/oauth2/v2.0/token",
    data=data, method="POST"
)
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
with urllib.request.urlopen(req, context=ctx) as r:
    token = json.loads(r.read())["access_token"]
print(f"Got token: {token[:30]}...")

for key, path in EHR_PATHS.items():
    bucket, name = key.split("/")
    out_dir = os.path.join(OUT, bucket)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{name}.pdf")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print(f"  [skip] {key}")
        continue
    enc = urllib.parse.quote(path, safe="")
    url = f"https://graph.microsoft.com/v1.0/drives/{DRIVE_ID}/root:/{enc}:/content"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=60) as r:
            data = r.read()
        with open(out_path, "wb") as fh:
            fh.write(data)
        print(f"  [ok]   {key}  ({len(data):>7} bytes)")
    except Exception as e:
        print(f"  [ERR]  {key}: {e}")

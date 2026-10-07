"""Claude CPT service: one Claude Code session per batch that fans charts out to Sonnet subagents.

POST /v1/jobs          multipart: file=<zip of PDFs>, rules=<CPT rules text, same as Gemini gets>, label=<optional>
GET  /v1/jobs/{id}     status, and per-file results once done
GET  /health

Every request needs header X-Token = $CLAUDE_CPT_TOKEN.
"""
import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile

HERE = Path(__file__).resolve().parent
JOBS_DIR = Path(os.environ.get("CLAUDE_CPT_JOBS_DIR", "/var/lib/claude-cpt/jobs"))
CLAUDE_BIN = os.environ.get("CLAUDE_BIN", "/root/.local/bin/claude")
TOKEN = os.environ.get("CLAUDE_CPT_TOKEN", "")
MODEL = os.environ.get("CLAUDE_CPT_MODEL", "sonnet")
CHARTS_PER_AGENT = int(os.environ.get("CLAUDE_CPT_CHARTS_PER_AGENT", "6"))
JOB_TIMEOUT = int(os.environ.get("CLAUDE_CPT_JOB_TIMEOUT", "2400"))
MAX_JOBS = threading.Semaphore(int(os.environ.get("CLAUDE_CPT_MAX_JOBS", "3")))
TOOL_FILES = ("xwalk.py", "pdftext.py", "playbook.md", "2025 crosswalk.xlsx")
ALLOWED_TOOLS = "Agent Read Write Bash(python3 xwalk.py:*) Bash(python3 pdftext.py:*)"

app = FastAPI()
_jobs = {}
_lock = threading.Lock()

SUBAGENT_PROMPT = """You are an expert anesthesia medical coder. For each chart listed in {batch}, assign the single main-line anesthesia CPT code (5-digit ASA code). Do not code block, line or add-on services (64xxx nerve blocks, 36xxx lines, 01968) as the main line.

Work only inside the current folder. Read fully first: rules.md (the production CPT rules; critical rules always win) and playbook.md (how to work and the crosswalk tools). Then code each chart following the playbook: read it with `python3 pdftext.py <pdf>`, view pages with no text layer using the Read tool, search the crosswalk, and verify your code plus at least one competing candidate with `python3 xwalk.py lookup`.

Write a JSON array to {out}, one object per chart: {{"case_id":"...","code":"5-digit code","confidence":"high|medium|low","procedure":"procedure performed, briefly","reasoning":"codes compared, why losers were rejected, crosswalk row relied on"}}. Write nothing else. Reply with one line confirming the file is written."""

ORCHESTRATOR_PROMPT = """You coordinate CPT coding for a batch of charts. The folder has {n} batch files: {batches}.
For EACH batch file, launch one subagent with the Agent tool (model: sonnet, run them in parallel, all in a single message). Give each subagent exactly the prompt in subagent_prompt.txt with {{batch}} replaced by its batch file name and {{out}} replaced by result_<N>.json (same N as the batch file).
Do not code any chart yourself. When every subagent has finished, check that every result_<N>.json exists; relaunch a subagent for any batch whose file is missing. Then reply with one line: DONE."""


def _check(token):
    if not TOKEN or token != TOKEN:
        raise HTTPException(status_code=401, detail="bad token")


def _run_claude(prompt, cwd, timeout):
    """One headless Claude Code run; returns (parsed json output or {}, error string)."""
    try:
        p = subprocess.run([CLAUDE_BIN, "-p", prompt, "--model", MODEL, "--output-format", "json",
                            "--allowedTools", ALLOWED_TOOLS],
                           cwd=cwd, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return {}, "timeout"
    try:
        return json.loads(p.stdout), ("" if p.returncode == 0 else f"exit {p.returncode}: {p.stderr[-300:]}")
    except json.JSONDecodeError:
        return {}, f"bad output (exit {p.returncode}): {(p.stderr or p.stdout)[-300:]}"


def _read_results(job_dir):
    out = {}
    for f in sorted(job_dir.glob("result_*.json")):
        try:
            for r in json.loads(f.read_text()):
                if isinstance(r, dict) and r.get("case_id"):
                    out[r["case_id"]] = r
        except (json.JSONDecodeError, OSError):
            pass
    return out


def _process(job_id):
    job = _jobs[job_id]
    job_dir = Path(job["dir"])
    with MAX_JOBS:
        job["status"] = "running"
        job["started"] = time.time()
        batches = sorted(job_dir.glob("batch_*.txt"), key=lambda p: int(re.search(r"\d+", p.name).group()))
        cost = 0.0
        orch, err = _run_claude(ORCHESTRATOR_PROMPT.format(n=len(batches), batches=", ".join(b.name for b in batches)),
                                job_dir, JOB_TIMEOUT)
        cost += float(orch.get("total_cost_usd") or 0)
        job["orchestrator_error"] = err

        # Backup: run any batch the orchestrator left without a result directly.
        missing = [b for b in batches if not (job_dir / b.name.replace("batch_", "result_").replace(".txt", ".json")).exists()]
        if missing:
            tpl = (job_dir / "subagent_prompt.txt").read_text()

            def one(b):
                out = b.name.replace("batch_", "result_").replace(".txt", ".json")
                return _run_claude(tpl.replace("{batch}", b.name).replace("{out}", out), job_dir, JOB_TIMEOUT // 2)

            with ThreadPoolExecutor(max_workers=len(missing)) as ex:
                for res, _ in ex.map(one, missing):
                    cost += float(res.get("total_cost_usd") or 0)
        job["rerun_batches"] = len(missing)

        res = _read_results(job_dir)
        manifest = json.loads((job_dir / "manifest.json").read_text())
        job["results"] = [{"filename": m["filename"], "case_id": cid,
                           "code": str(res.get(cid, {}).get("code", "")).strip(),
                           "confidence": res.get(cid, {}).get("confidence", ""),
                           "procedure": res.get(cid, {}).get("procedure", ""),
                           "reasoning": res.get(cid, {}).get("reasoning", "")}
                          for cid, m in manifest.items()]
        job["cost_usd"] = round(cost, 4)
        job["duration_s"] = round(time.time() - job["started"])
        job["status"] = "done" if all(r["code"] for r in job["results"]) else "partial"
        (job_dir / "job.json").write_text(json.dumps({k: v for k, v in job.items() if k != "dir"}, indent=1))


@app.get("/health")
def health():
    return {"ok": True, "jobs_running": sum(1 for j in _jobs.values() if j["status"] == "running")}


@app.post("/v1/jobs")
async def create_job(file: UploadFile = File(...), rules: str = Form(...), label: str = Form(""),
                     x_token: str = Header("")):
    _check(x_token)
    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_DIR / job_id
    (job_dir / "pdfs").mkdir(parents=True)
    zpath = job_dir / "in.zip"
    with open(zpath, "wb") as f:
        shutil.copyfileobj(file.file, f)
    manifest = {}
    with zipfile.ZipFile(zpath) as z:
        names = sorted(n for n in z.namelist() if n.lower().endswith(".pdf") and "__MACOSX" not in n)
        for i, n in enumerate(names, 1):
            cid = f"C{i:03d}"
            (job_dir / "pdfs" / f"{cid}.pdf").write_bytes(z.read(n))
            manifest[cid] = {"filename": n.split("/")[-1]}
    zpath.unlink()
    if not manifest:
        shutil.rmtree(job_dir)
        raise HTTPException(status_code=400, detail="zip has no PDFs")
    for t in TOOL_FILES:
        shutil.copy(HERE / t, job_dir / t)
    (job_dir / "rules.md").write_text(rules)
    (job_dir / "subagent_prompt.txt").write_text(SUBAGENT_PROMPT)
    (job_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    ids = list(manifest)
    for b, start in enumerate(range(0, len(ids), CHARTS_PER_AGENT), 1):
        lines = [f"- {cid}: pdfs/{cid}.pdf" for cid in ids[start:start + CHARTS_PER_AGENT]]
        (job_dir / f"batch_{b}.txt").write_text("\n".join(lines) + "\n")
    with _lock:
        _jobs[job_id] = {"id": job_id, "label": label, "status": "queued", "n_charts": len(ids),
                         "created": time.time(), "dir": str(job_dir)}
    threading.Thread(target=_process, args=(job_id,), daemon=True).start()
    return {"job_id": job_id, "n_charts": len(ids)}


@app.get("/v1/jobs/{job_id}")
def get_job(job_id: str, x_token: str = Header("")):
    _check(x_token)
    job = _jobs.get(job_id)
    if not job:
        saved = JOBS_DIR / job_id / "job.json"
        if saved.exists():
            return json.loads(saved.read_text())
        raise HTTPException(status_code=404, detail="unknown job")
    return {k: v for k, v in job.items() if k != "dir"}

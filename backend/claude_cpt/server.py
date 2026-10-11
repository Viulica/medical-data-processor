"""Claude coding service: the charts of a job are split into batches and each batch is coded by its own
headless Claude Code session (no orchestrator); a batch is rerun once if its results are missing or malformed.

POST /v1/jobs          multipart: file=<zip of PDFs>, rules=<the exact prompt the Gemini one-shot call gets>,
                       task=cpt|icd|cpticd, guidance=<optional JSON {pdf filename: per-chart prompt addendum}>,
                       rules_icd=<task cpticd only: the exact ICD prompt>, cpt_guidance=<cpticd: 1 = append the
                       predicted-CPT block to the ICD prompt, as the serial CPT->ICD mode does>, label=<optional>
                       cpticd codes CPT and ICD in one pass, so every chart is read once.
GET  /v1/jobs/{id}     status, and per-file results once done
POST /v1/complete      JSON {model: haiku|sonnet|<model id>, system, content: [{"type":"text","text":...} |
                       {"type":"image","data":<base64 jpeg>}]}: one headless one-turn Claude call without tools
                       (chart field extraction); returns {text, cost_usd, usage}.
GET  /health

Routing (CLAUDE_CPT_ROUTE, on by default): jobs with a CPT code are coded by Haiku first; a chart keeps Haiku's
CPT and ICD only when the CPT is one of the simple codes in CLAUDE_CPT_SIMPLE, every other chart is recoded by
the main model (CLAUDE_CPT_MODEL).

Every request needs header X-Token = $CLAUDE_CPT_TOKEN.
"""
import asyncio
import hashlib
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
import sys

from fastapi import Body, FastAPI, File, Form, Header, HTTPException, UploadFile

HERE = Path(__file__).resolve().parent
JOBS_DIR = Path(os.environ.get("CLAUDE_CPT_JOBS_DIR", "/var/lib/claude-cpt/jobs"))
CLAUDE_BIN = os.environ.get("CLAUDE_BIN", "/root/.local/bin/claude")
TOKEN = os.environ.get("CLAUDE_CPT_TOKEN", "")
MODEL = os.environ.get("CLAUDE_CPT_MODEL", "sonnet")
HAIKU = os.environ.get("CLAUDE_CPT_HAIKU_MODEL", "claude-haiku-5-5")
SONNET = os.environ.get("CLAUDE_CPT_SONNET_MODEL", "claude-sonnet-5-5")
ROUTE = os.environ.get("CLAUDE_CPT_ROUTE", "1") != "0"
# CPT codes Haiku codes as well as Sonnet (colonoscopy/EGD, cataract, labor epidural): those charts keep Haiku's
# CPT and ICD. Tested on 252 charts: same CPT accuracy as all-Sonnet, ICD1 179 vs 180, ~35% less usage.
SIMPLE_CPT = set(os.environ.get("CLAUDE_CPT_SIMPLE", "00811,00812,00813,00731,00732,00140,00142,01967").split(","))
CHARTS_PER_AGENT = int(os.environ.get("CLAUDE_CPT_CHARTS_PER_AGENT", "6"))
JOB_TIMEOUT = int(os.environ.get("CLAUDE_CPT_JOB_TIMEOUT", "2400"))
# Claude Code sessions running at once, across all jobs.
SESSIONS = threading.Semaphore(int(os.environ.get("CLAUDE_CPT_MAX_SESSIONS", "12")))
# Seconds the first batch of a job runs alone so its system prompt is cached before the other batches start.
WARMUP_S = float(os.environ.get("CLAUDE_CPT_WARMUP_S", "12"))
# One-turn extraction calls running at once (each is a short claude process).
COMPLETE_SESSIONS = int(os.environ.get("CLAUDE_COMPLETE_MAX_SESSIONS", "10"))
COMPLETE_TIMEOUT = int(os.environ.get("CLAUDE_COMPLETE_TIMEOUT", "240"))
COMPLETE_DIR = JOBS_DIR.parent / "complete"
CROSSWALK = HERE / "2025 crosswalk.xlsx" if (HERE / "2025 crosswalk.xlsx").exists() else HERE.parent / "2025 crosswalk.xlsx"
ICD_FILES = ("icd10.py", "icd10cm_fy26_fy27.tsv.gz")
TASKS = {
    "cpt": {"files": ("xwalk.py", "pdftext.py", "playbook.md"), "keys": ("code",),
            "tools": "Read Write Bash(python3 xwalk.py:*) Bash(python3 pdftext.py:*)"},
    "icd": {"files": ("pdftext.py", "playbook_icd.md") + ICD_FILES, "keys": ("ICD1",),
            "tools": "Read Write Bash(python3 pdftext.py:*) Bash(python3 icd10.py:*)"},
    "cpticd": {"files": ("xwalk.py", "pdftext.py", "playbook.md", "playbook_icd.md", "cptguide.py") + ICD_FILES,
               "keys": ("code", "ICD1"),
               "tools": "Read Write Bash(python3 xwalk.py:*) Bash(python3 pdftext.py:*) Bash(python3 cptguide.py:*) "
                        "Bash(python3 icd10.py:*)"},
}
NO_TOOLS = "Agent ListAgents ScheduleWakeup SendMessage"
# Lean sessions: only the tools the coders use, no MCP connectors / skills, and a short system prompt (system.md,
# written per job) instead of Claude Code's default one. system.md also carries the job's rules and playbooks word
# for word, so sessions don't Read them (no line numbers, no extra turns) and every batch of a job starts from the
# same prompt prefix, which the prompt cache can share. Cuts the context each session starts with from ~22k tokens.
LEAN = os.environ.get("CLAUDE_CPT_LEAN", "1") != "0"
LEAN_ARGS = ["--tools", "Read,Write,Bash", "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
             "--disable-slash-commands", "--no-session-persistence", "--system-prompt-file", "system.md"]
LEAN_PREAMBLE = ("You are a medical coding worker run headless by a script. Do exactly the task in the user message, "
                 "inside the current folder: read files and PDF pages with the Read tool, write results with the Write "
                 "tool, and use Bash only for the python3 tools the task names. Never ask questions; finish the task.\n\n"
                 "The files below are reproduced word for word from the job folder; they are already loaded, so never "
                 "Read them.")
SHELL_NOTE = ("Run each tool as one plain command from the current folder (no cd, no pipes, no && chains, no cat/head/grep); "
              "read files with the Read tool.")

# LiteParse (pip install liteparse) gives layout-aware text per page. OCR stays off: pages without a text
# layer are flagged so the agent views them as images itself.
LIT_BIN = os.environ.get("LIT_BIN") or shutil.which("lit") or str(Path(sys.executable).parent / "lit")
NO_TEXT = "[NO TEXT LAYER - view this page as an image with the Read tool]"

app = FastAPI()
_jobs = {}
_lock = threading.Lock()

SUBAGENT_PROMPTS = {
    "cpt": """You are an expert anesthesia medical coder. For each chart listed in {batch}, assign the single main-line anesthesia CPT code (5-digit ASA code). Do not code block, line or add-on services (64xxx nerve blocks, 36xxx lines, 01968) as the main line.

Work only inside the current folder. Read fully first: rules.md (the EXACT prompt the production one-shot model gets: follow every rule in it; only its output format is replaced by the one below) and playbook.md (how to work and the crosswalk tools). Then code each chart following the playbook: the text of every chart in your batch is already extracted in the batch's _text.md file (named in the batch file; read it fully, using offset/limit if it is long); view any page marked NO TEXT LAYER from that chart's PDF with the Read tool; search the crosswalk; and verify your code plus at least one competing candidate with `python3 xwalk.py lookup` (it takes several codes at once). {shell}

Write a JSON array to {out}, one object per chart: {{"case_id":"...","code":"5-digit code","confidence":"high|medium|low","procedure":"procedure performed, briefly","reasoning":"codes compared, why losers were rejected, crosswalk row relied on"}}. Write nothing else. Reply with one line confirming the file is written.""",
    "icd": """You are an expert anesthesia medical coder. For each chart listed in {batch}, assign the ICD-10-CM diagnosis codes ICD1 (primary) to ICD4.

Work only inside the current folder. Read fully first: rules.md (the EXACT prompt the production one-shot model gets: follow every rule in it; only its output format is replaced by the one below) and playbook_icd.md (how to work and the ICD-10 tools). If a chart's line names a guidance file, read it: it is the per-chart addendum the production model gets for that chart and is part of the prompt. Then code each chart following the playbook: the text of every chart in your batch is already extracted in the batch's _text.md file (named in the batch file; read it fully, using offset/limit if it is long); view any page marked NO TEXT LAYER from that chart's PDF with the Read tool; and verify EVERY code with `python3 icd10.py check --dos <date of service> <code> [<code> ...]` (one call per chart, all its codes at once). {shell}

Write a JSON array to {out}, one object per chart: {{"case_id":"...","ICD1":"code","ICD1_Reasoning":"1-2 sentences","ICD2":"code or empty","ICD2_Reasoning":"","ICD3":"code or empty","ICD3_Reasoning":"","ICD4":"code or empty","ICD4_Reasoning":"","confidence":"high|medium|low"}}. Codes in dotted form (e.g. K63.5). Write nothing else. Reply with one line confirming the file is written.""",
    "cpticd": """You are an expert anesthesia medical coder. For each chart listed in {batch}, assign (A) the single main-line anesthesia CPT code (5-digit ASA code; never a block, line or add-on service such as 64xxx nerve blocks, 36xxx lines or 01968) and (B) the ICD-10-CM diagnosis codes ICD1 (primary) to ICD4. Read each chart once and do both.

Work only inside the current folder. Read fully first: rules.md (the EXACT CPT prompt the production one-shot model gets), rules_icd.md (the EXACT ICD prompt the production one-shot model gets), playbook.md (how to code CPT and the crosswalk tools) and playbook_icd.md (how to code ICD and the ICD-10 tools). Follow every rule in both prompts; only their output formats are replaced by the one below. The text of every chart in your batch is already extracted in the batch's _text.md file (named in the batch file; read it fully, using offset/limit if it is long); view any page marked NO TEXT LAYER from that chart's PDF with the Read tool.
For each chart: first code the CPT following playbook.md (search the crosswalk; verify your code plus at least one competing candidate with `python3 xwalk.py lookup`, which takes several codes at once). {guide}Then code the ICDs following playbook_icd.md and verify EVERY ICD code with `python3 icd10.py check --dos <date of service> <code> [<code> ...]` (one call per chart, all its codes at once). {shell}

Write a JSON array to {out}, one object per chart: {{"case_id":"...","code":"5-digit CPT code","confidence":"high|medium|low","procedure":"procedure performed, briefly","reasoning":"CPT codes compared, why losers were rejected, crosswalk row relied on","ICD1":"code","ICD1_Reasoning":"1-2 sentences","ICD2":"code or empty","ICD2_Reasoning":"","ICD3":"code or empty","ICD3_Reasoning":"","ICD4":"code or empty","ICD4_Reasoning":"","icd_confidence":"high|medium|low"}}. ICD codes in dotted form (e.g. K63.5). Write nothing else. Reply with one line confirming the file is written.""",
}
CPT_GUIDE_STEP = ("The production ICD prompt ends with a predicted-CPT block for the chart: once your CPT codes for the batch "
                  "are final, print those blocks with `python3 cptguide.py <code> [<code> ...]` (one call for the batch) "
                  "and treat each chart's block as the end of rules_icd.md for that chart. ")



def _check(token):
    if not TOKEN or token != TOKEN:
        raise HTTPException(status_code=401, detail="bad token")


def _run_claude(prompt, cwd, timeout, tools, model=None):
    """One headless Claude Code run; returns (parsed json output or {}, error string)."""
    try:
        p = subprocess.run([CLAUDE_BIN, "-p", prompt, "--model", model or MODEL, "--output-format", "json",
                            "--allowedTools", tools, "--disallowedTools", NO_TOOLS] + (LEAN_ARGS if LEAN else []),
                           cwd=cwd, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return {}, "timeout"
    try:
        return json.loads(p.stdout), ("" if p.returncode == 0 else f"exit {p.returncode}: {p.stderr[-300:]}")
    except json.JSONDecodeError:
        return {}, f"bad output (exit {p.returncode}): {(p.stderr or p.stdout)[-300:]}"


def _page_texts(pdf):
    """Per-page text: LiteParse when available, PyMuPDF as a fallback."""
    try:
        p = subprocess.run([LIT_BIN, "parse", str(pdf), "--no-ocr", "--format", "json", "-q"],
                           capture_output=True, text=True, timeout=120)
        pages = json.loads(p.stdout)["pages"]
        return [pg.get("text", "") for pg in sorted(pages, key=lambda x: x["page"])]
    except Exception:
        import fitz
        return [pg.get_text() for pg in fitz.open(pdf)]


def _chart_text(cid, pdf):
    pages = _page_texts(pdf)
    out = [f"########## CHART {cid} (pdfs/{cid}.pdf, {len(pages)} pages) ##########"]
    for i, t in enumerate(pages, 1):
        # LiteParse pads columns with runs of spaces; two spaces keep the columns apart for a third of the text.
        t = re.sub(r"[ \t]+\n", "\n", re.sub(r"[ \t]{2,}", "  ", t)).strip()
        out.append(f"===== {cid} PAGE {i}/{len(pages)} =====")
        out.append(t if len(t) > 80 else NO_TEXT)
    return "\n".join(out)


def _load_rows(path):
    """Rows of one results file; tolerates text around the JSON array. None if unreadable."""
    try:
        txt = path.read_text()
    except OSError:
        return None
    for cand in (txt, txt[txt.find("["):txt.rfind("]") + 1]):
        try:
            data = json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(data, dict):
            data = data.get("results", [])
        if isinstance(data, list):
            return [r for r in data if isinstance(r, dict) and r.get("case_id")]
    return None


def _batch_ok(job_dir, batch, keys):
    """True when the batch's results file has every chart with every required field."""
    rows = _load_rows(job_dir / batch.name.replace("batch_", "result_").replace(".txt", ".json"))
    if rows is None:
        return False
    got = {r["case_id"]: r for r in rows}
    want = re.findall(r"^- (C\d+):", batch.read_text(), flags=re.M)
    return all(str(got.get(c, {}).get(k, "")).strip() for c in want for k in keys)


def _usage(out, acc):
    for model, u in (out.get("modelUsage") or {}).items():
        a = acc.setdefault(model, {})
        for k in ("inputTokens", "outputTokens", "cacheReadInputTokens", "cacheCreationInputTokens"):
            a[k] = a.get(k, 0) + int(u.get(k, 0) or 0)


def _process(job_id):
    job = _jobs[job_id]
    job_dir = Path(job["dir"])
    job["status"] = "running"
    job["started"] = time.time()
    batches = sorted(job_dir.glob("batch_*.txt"), key=lambda p: int(re.search(r"\d+", p.name).group()))
    spec = TASKS[job["task"]]
    tpl = (job_dir / "subagent_prompt.txt").read_text()
    usage, errors = {}, []
    stats = {"sessions": 0, "reruns": 0, "cost": 0.0}
    lock = threading.Lock()

    def one(b, model):
        out = b.name.replace("batch_", "result_").replace(".txt", ".json")
        for attempt in (1, 2):
            with SESSIONS:
                res, err = _run_claude(tpl.replace("{batch}", b.name).replace("{out}", out), job_dir, JOB_TIMEOUT,
                                       spec["tools"], model)
            with lock:
                stats["sessions"] += 1
                stats["reruns"] += attempt - 1
                stats["cost"] += float(res.get("total_cost_usd") or 0)
                _usage(res, usage)
                if err:
                    errors.append(f"{b.name} try {attempt}: {err}")
            if _batch_ok(job_dir, b, spec["keys"]):
                return

    def run_round(round_batches, model):
        # Every batch of a job starts from the same system prompt (~45k tokens of rules). Sessions that start
        # together each pay to write it to the prompt cache; start one first so the rest read it from the cache.
        with ThreadPoolExecutor(max_workers=max(1, len(round_batches))) as ex:
            futures = [ex.submit(one, round_batches[0], model)] if round_batches else []
            if len(round_batches) > 1:
                time.sleep(WARMUP_S)
                futures += [ex.submit(one, b, model) for b in round_batches[1:]]
            for f in futures:
                f.result()

    def collect(prefix):
        res = {}
        for f in sorted(job_dir.glob(f"{prefix}*.json")):
            for r in _load_rows(f) or []:
                res[r["case_id"]] = r
        return res

    manifest = json.loads((job_dir / "manifest.json").read_text())
    routed = ROUTE and "code" in spec["keys"]
    run_round(batches, HAIKU if routed else MODEL)
    res = collect("result_")
    if routed:
        # Charts Haiku coded with a simple CPT keep its codes; the rest (and any it missed) go to the main model.
        hard = [cid for cid in manifest if str(res.get(cid, {}).get("code", "")).strip().zfill(5) not in SIMPLE_CPT]
        job["routing"] = {"haiku_kept": len(manifest) - len(hard), "escalated": len(hard)}
        esc = []
        for n, start in enumerate(range(0, len(hard), CHARTS_PER_AGENT), 1):
            esc.append(_write_batch(job_dir, manifest, hard[start:start + CHARTS_PER_AGENT], f"s{n}"))
        run_round(esc, MODEL)
        res.update({cid: dict(r, model="sonnet") for cid, r in collect("result_s").items()})
        for cid in manifest:
            if cid in res and "model" not in res[cid]:
                res[cid]["model"] = "haiku"

    job["results"] = [dict({k: v for k, v in res.get(cid, {}).items() if k != "case_id"},
                           filename=m["filename"], case_id=cid) for cid, m in manifest.items()]
    job.update(cost_usd=round(stats["cost"], 4), sessions=stats["sessions"], rerun_batches=stats["reruns"],
               usage=usage, errors=errors[-10:], duration_s=round(time.time() - job["started"]))
    job["status"] = "done" if all(str(r.get(k, "")).strip() for r in job["results"] for k in spec["keys"]) else "partial"
    (job_dir / "job.json").write_text(json.dumps({k: v for k, v in job.items() if k != "dir"}, indent=1))


def _write_batch(job_dir, manifest, chunk, name):
    lines = [f"- {cid}: pdfs/{cid}.pdf" + (f" (guidance: {manifest[cid]['guidance']})" if manifest[cid].get("guidance") else "")
             for cid in chunk]
    lines.append(f"Text of all these charts, already extracted page by page: batch_{name}_text.md")
    (job_dir / f"batch_{name}.txt").write_text("\n".join(lines) + "\n")
    (job_dir / f"batch_{name}_text.md").write_text(
        "\n\n".join((job_dir / "texts" / f"{cid}.md").read_text() for cid in chunk) + "\n")
    return job_dir / f"batch_{name}.txt"


@app.get("/health")
def health():
    return {"ok": True, "jobs_running": sum(1 for j in _jobs.values() if j["status"] == "running")}


@app.post("/v1/jobs")
async def create_job(file: UploadFile = File(...), rules: str = Form(...), label: str = Form(""),
                     task: str = Form("cpt"), guidance: str = Form(""), rules_icd: str = Form(""),
                     cpt_guidance: str = Form("1"), x_token: str = Header("")):
    _check(x_token)
    if task not in TASKS:
        raise HTTPException(status_code=400, detail=f"unknown task {task}")
    if task == "cpticd" and not rules_icd.strip():
        raise HTTPException(status_code=400, detail="task cpticd needs rules_icd")
    per_chart = json.loads(guidance) if guidance.strip() else {}
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
    for t in TASKS[task]["files"]:
        shutil.copy(HERE / t, job_dir / t)
    if "xwalk.py" in TASKS[task]["files"]:
        shutil.copy(CROSSWALK, job_dir / "2025 crosswalk.xlsx")
    (job_dir / "rules.md").write_text(rules)
    prompt = SUBAGENT_PROMPTS[task].replace("{shell}", SHELL_NOTE)
    if task == "cpticd":
        (job_dir / "rules_icd.md").write_text(rules_icd)
        # The ICD playbook is written for ICD-only jobs: point it at this job's ICD prompt and CPT block.
        pb = (job_dir / "playbook_icd.md").read_text()
        (job_dir / "playbook_icd.md").write_text(
            pb.replace("rules.md", "rules_icd.md").replace("the chart's guidance file", "the chart's predicted-CPT block")
            .replace("the guidance file", "the predicted-CPT block"))
        prompt = prompt.replace("{guide}", CPT_GUIDE_STEP if cpt_guidance.strip() not in ("0", "false", "") else "")
    if LEAN:
        parts = [LEAN_PREAMBLE]
        for name in ("playbook.md", "playbook_icd.md", "rules.md", "rules_icd.md"):
            f = job_dir / name
            if f.exists():
                parts.append(f"\n\n==================== FILE: {name} ====================\n{f.read_text()}")
        (job_dir / "system.md").write_text("".join(parts))
        prompt = prompt.replace("Read fully first:", "Already loaded in your system prompt, word for word (do not Read them):")
    (job_dir / "subagent_prompt.txt").write_text(prompt)
    for cid, m in manifest.items():
        text = per_chart.get(m["filename"], "")
        if text.strip():
            (job_dir / "guidance").mkdir(exist_ok=True)
            (job_dir / "guidance" / f"{cid}.md").write_text(text)
            m["guidance"] = f"guidance/{cid}.md"
    (job_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    ids = list(manifest)
    (job_dir / "texts").mkdir()
    with ThreadPoolExecutor(max_workers=8) as ex:
        for cid, text in zip(ids, ex.map(lambda c: _chart_text(c, job_dir / "pdfs" / f"{c}.pdf"), ids)):
            (job_dir / "texts" / f"{cid}.md").write_text(text)
    for b, start in enumerate(range(0, len(ids), CHARTS_PER_AGENT), 1):
        _write_batch(job_dir, manifest, ids[start:start + CHARTS_PER_AGENT], str(b))
    with _lock:
        _jobs[job_id] = {"id": job_id, "label": label, "task": task, "status": "queued", "n_charts": len(ids),
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


_complete_sem = None


@app.post("/v1/complete")
async def complete(body: dict = Body(...), x_token: str = Header("")):
    """One-turn call without tools: the system prompt plus text and images in one user message, like a plain
    API call. Async so that waiting calls never hold the worker threads the job endpoints need."""
    global _complete_sem
    _check(x_token)
    model = {"haiku": HAIKU, "sonnet": SONNET}.get(body.get("model", "haiku"), body.get("model"))
    content = []
    for blk in body.get("content") or []:
        if blk.get("type") == "image":
            content.append({"type": "image", "source": {"type": "base64", "media_type": blk.get("media_type", "image/jpeg"),
                                                        "data": blk["data"]}})
        elif blk.get("text"):
            content.append({"type": "text", "text": blk["text"]})
    if not content:
        raise HTTPException(status_code=400, detail="empty content")
    # The system prompt is the same for every chart of a template: a stable file keeps it in the prompt cache.
    system = body.get("system") or "You extract data from medical documents. Reply with JSON only."
    COMPLETE_DIR.mkdir(parents=True, exist_ok=True)
    sys_file = COMPLETE_DIR / f"sys_{hashlib.sha1(system.encode()).hexdigest()[:16]}.md"
    if not sys_file.exists():
        sys_file.write_text(system)
    msg = json.dumps({"type": "user", "message": {"role": "user", "content": content}}) + "\n"
    if _complete_sem is None:
        _complete_sem = asyncio.Semaphore(COMPLETE_SESSIONS)
    async with _complete_sem:
        proc = await asyncio.create_subprocess_exec(
            CLAUDE_BIN, "-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
            "--model", model, "--tools", "", "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
            "--disable-slash-commands", "--no-session-persistence", "--system-prompt-file", str(sys_file),
            cwd=str(COMPLETE_DIR), stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)
        try:
            out, err = await asyncio.wait_for(proc.communicate(msg.encode()), COMPLETE_TIMEOUT)
        except asyncio.TimeoutError:
            proc.kill()
            raise HTTPException(status_code=504, detail="timeout")
    result = None
    for line in out.decode(errors="replace").splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("type") == "result":
            result = d
    if not result or result.get("is_error") or not result.get("result"):
        detail = (result or {}).get("result") or err.decode(errors="replace")[-300:] or f"exit {proc.returncode}"
        raise HTTPException(status_code=502, detail=f"claude failed: {detail}")
    try:
        u = (result.get("modelUsage") or {}).get(model, {})
        with open(COMPLETE_DIR / "usage.jsonl", "a") as f:
            f.write(json.dumps({"t": round(time.time()), "model": model, "cost": result.get("total_cost_usd"),
                                "images": sum(b["type"] == "image" for b in content),
                                **{k: u.get(k, 0) for k in ("inputTokens", "outputTokens", "cacheReadInputTokens",
                                                            "cacheCreationInputTokens")}}) + "\n")
    except OSError:
        pass
    return {"text": result["result"], "model": model, "cost_usd": result.get("total_cost_usd"),
            "usage": result.get("modelUsage")}


#!/usr/bin/env python3
"""
CPT prediction — AGENT MODE.

Mirrors production's context (base CPT prompt + full anesthesia code list + group
instruction template) and ADDS three ASA-Crosswalk tools. The model reads the PDF,
applies the rules/template first, then uses the crosswalk to investigate anything
uncertain, looping until it converges, then emits a final JSON answer.

Default model: google/gemini-3.5-flash.
Usage: python3 cpt_agent.py <pdf> [--model M] [--pages N]
"""
import sys, os, json, argparse, time, ssl
import numpy as np, pandas as pd, requests

# This module now lives in backend/. Resolve paths relative to this file so it
# works on any host (no hardcoded absolute path).
HERE = os.path.dirname(os.path.abspath(__file__))          # .../backend
ROOT = os.path.dirname(HERE)                                # repo root
XL = os.path.join(HERE, "2025 crosswalk.xlsx")             # backend/2025 crosswalk.xlsx
EMB_CACHE = os.path.join(HERE, "xwalk_emb.npz")            # backend/xwalk_emb.npz

def load_env():
    # Best-effort load of backend/.env; on the server env vars are usually set
    # directly, so a missing .env must not crash import.
    env_path = os.path.join(HERE, ".env")
    if not os.path.exists(env_path):
        return
    for line in open(env_path):
        line=line.strip()
        if line and not line.startswith("#") and "=" in line:
            k,v=line.split("=",1); os.environ.setdefault(k, v.strip().strip('"').strip("'"))
load_env()
OR_KEY=os.environ.get("OPENROUTER_API_KEY"); OAI_KEY=os.environ.get("OPENAI_API_KEY")
_CTX=ssl.create_default_context(); _CTX.check_hostname=False; _CTX.verify_mode=ssl.CERT_NONE

# ---- crosswalk table ----
_DF=None
def xwalk_df():
    global _DF
    if _DF is None:
        d=pd.read_excel(XL,sheet_name=0,header=8)
        d=d.rename(columns={'CPT Procedure Code':'surg','CPT Procedure Descriptor':'surg_desc',
                            'CPT Anesthesia Code':'anes','CPT Anesthesia Descriptor':'anes_desc',
                            'Base Unit Value':'base_units','Time':'time_rule','Alternates':'alternates',
                            'Comment':'comment','Instructi/Text':'instruction'})
        d=d[d['surg_desc'].notna()].reset_index(drop=True)
        d['surg']=d['surg'].astype(str).str.replace(r'\.0$','',regex=True)
        d['anes']=d['anes'].astype(str).str.replace(r'\.0$','',regex=True)
        d['sd']=d['surg_desc'].str.lower()
        _DF=d
    return _DF

def _clean(v):
    """stringify a cell, dropping NaN/empty/0.0 flag noise."""
    if v is None: return ""
    s=str(v).strip()
    if s.lower() in ("nan","none","0.0",""): return ""
    return s

_ANES_DESC=None
def anes_desc_map():
    """anesthesia code -> its official descriptor (first non-placeholder occurrence)."""
    global _ANES_DESC
    if _ANES_DESC is None:
        d=xwalk_df(); m={}
        for _,r in d.iterrows():
            c=r['anes']; dsc=_clean(r.get('anes_desc'))
            if c and dsc and c not in m and 'NOT A PRIMARY' not in dsc.upper() and 'NOT TYPICALLY' not in dsc.upper():
                m[c]=dsc
        _ANES_DESC=m
    return _ANES_DESC

def _row(r, with_instruction=True):
    """Full crosswalk row for the agent — includes ALL useful columns, not just code+desc."""
    out={"surg":r['surg'],"anes":r['anes'],"surg_desc":_clean(r['surg_desc'])[:140]}
    bu=_clean(r.get('base_units')); alt=_clean(r.get('alternates'))
    com=_clean(r.get('comment')); ins=_clean(r.get('instruction'))
    if bu: out["base_units"]=bu
    if alt:
        # expand each alternate code to "code: descriptor" so the agent can reason without extra lookups
        dm=anes_desc_map()
        out["alternate_codes"]=[{"anes":c.strip(),"desc":dm.get(c.strip(),"")[:120]}
                                for c in alt.split(",") if c.strip()]
    if with_instruction:
        if com: out["comment"]=com[:240]
        if ins: out["instruction"]=ins[:240]
    return out

# Embeddings: gemini-embedding-2 via OpenRouter (matches the validated harness).
# The pre-built corpus cache is xwalk_emb_gemini.npz (3072-dim). No OpenAI dependency.
EMB_MODEL=os.environ.get("XWALK_EMBED_MODEL","google/gemini-embedding-2")
GEMINI_EMB_CACHE=os.path.join(HERE,"xwalk_emb_gemini.npz")
def _embed(texts):
    import urllib.request
    out=[]
    for i in range(0,len(texts),100):
        body=json.dumps({"model":EMB_MODEL,"input":texts[i:i+100]}).encode()
        req=urllib.request.Request("https://openrouter.ai/api/v1/embeddings",data=body,
            headers={"Authorization":f"Bearer {OR_KEY}","Content-Type":"application/json"})
        r=json.load(urllib.request.urlopen(req,context=_CTX))
        out.extend([d["embedding"] for d in r["data"]])
    return np.array(out,dtype=np.float32)
_EMB=None
def xwalk_emb():
    global _EMB
    if _EMB is None:
        d=xwalk_df()
        # Prefer the pre-built gemini cache (shipped in repo); fall back to the old one.
        for cache in (GEMINI_EMB_CACHE, EMB_CACHE):
            if os.path.exists(cache):
                z=np.load(cache)
                if len(z["emb"])==len(d): _EMB=z["emb"]; return _EMB
        e=_embed(d['surg_desc'].tolist()); e=e/(np.linalg.norm(e,axis=1,keepdims=True)+1e-9)
        np.savez(GEMINI_EMB_CACHE,emb=e); _EMB=e
    return _EMB

# ---- 3 tools ----
def t_string_search(terms):
    d=xwalk_df(); m=pd.Series(True,index=d.index)
    for t in terms: m&=d['sd'].str.contains(str(t).lower(),regex=False,na=False)
    return [_row(r) for _,r in d[m].head(20).iterrows()]
def t_embedding_search(query,k=10):
    d=xwalk_df(); emb=xwalk_emb()
    q=_embed([query])[0]; q=q/(np.linalg.norm(q)+1e-9); sims=emb@q; idx=np.argsort(-sims)[:k]
    rows=[]
    for i in idx:
        row=_row(d.iloc[i]); row["sim"]=round(float(sims[i]),3); rows.append(row)
    return rows
def t_cpt_lookup(anes_code):
    d=xwalk_df(); code=str(anes_code).zfill(5); sub=d[d['anes']==code]
    if sub.empty: return {"anes":code,"found":False,"note":"no surgical procedures map to this code"}
    desc=_clean(sub.iloc[0]['anes_desc']) if 'anes_desc' in sub.columns else ""
    # surface any coding comments/instructions attached to rows that map to this code
    notes=[]
    for _,r in sub.iterrows():
        for col in ('comment','instruction'):
            v=_clean(r.get(col))
            if v and v not in notes: notes.append(v[:240])
    return {"anes":code,"found":True,"anes_descriptor":desc[:240],"num_surgical_procedures":len(sub),
            "coding_notes":notes[:6],
            "examples":[f"{r['surg']}: {_clean(r['surg_desc'])[:80]}" for _,r in sub.head(12).iterrows()]}

def t_grep_search(pattern, whole_word=False, max_results=25):
    """Regex grep over crosswalk SURGICAL descriptors (like grep -E). Supports
    alternation (a|b), anchors, char classes. Falls back to literal substring if
    the pattern is not valid regex."""
    import re as _re
    d=xwalk_df(); pat=str(pattern)
    if whole_word: pat=r"\b(?:%s)\b"%pat
    try:
        rx=_re.compile(pat,_re.IGNORECASE)
        mask=d["surg_desc"].astype(str).apply(lambda s: bool(rx.search(s)))
    except _re.error:
        mask=d["sd"].str.contains(pat.lower(),regex=False,na=False)
    hits=d[mask]
    return {"pattern":pattern,"n_matches":int(mask.sum()),
            "results":[_row(r) for _,r in hits.head(max_results).iterrows()]}

SERPER_KEY=os.environ.get("SERPER_API_KEY","5e3d6df8e4bf3b187a80d46b0adff922efdb0862")
def t_web_search(query, num=5):
    """Google web search via Serper — works for ANY model (incl. self-hosted vLLM,
    which cannot use OpenRouter's server-side web_search). Returns top results +
    answer box for grounding the CPT choice."""
    try:
        r=requests.post("https://google.serper.dev/search",
            headers={"X-API-KEY":SERPER_KEY,"Content-Type":"application/json"},
            json={"q":query,"num":num},timeout=25)
        r.raise_for_status(); d=r.json()
    except Exception as e:
        return {"query":query,"error":str(e)[:120]}
    out={"query":query}
    if d.get("answerBox"):
        ab=d["answerBox"]; out["answer_box"]=(ab.get("answer") or ab.get("snippet") or "")[:400]
    if d.get("knowledgeGraph"):
        out["knowledge_graph"]=(d["knowledgeGraph"].get("description") or "")[:300]
    out["results"]=[{"title":(o.get("title") or "")[:120],"snippet":(o.get("snippet") or "")[:280],
                     "link":(o.get("link") or "")[:120]} for o in (d.get("organic") or [])[:num]]
    return out

# Function-tool specs for the local (vLLM) toolset: real regex grep + Serper web.
_GREP_TOOL_SPEC={"type":"function","function":{"name":"crosswalk_grep",
   "description":"GREP the ASA Crosswalk surgical descriptors with a real REGEX (grep -E): alternation 'knee|patella', 'arthroscop(y|ic)', 'excis(e|ion).*(mass|lesion|lipoma)'. Case-insensitive. More powerful than plain keyword match. Returns matching full crosswalk rows + total match count.",
   "parameters":{"type":"object","properties":{"pattern":{"type":"string"},"whole_word":{"type":"boolean"}},"required":["pattern"]}}}
_WEB_TOOL_SPEC={"type":"function","function":{"name":"web_search",
   "description":"Google web search for the standard anesthesia (ASA) CPT code of a documented procedure (e.g. 'anesthesia CPT code for total knee arthroplasty'). Returns top results + answer box. Prefer a code supported by BOTH web AND crosswalk.",
   "parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}}

TOOLS_SPEC=[
 # web search FIRST so the model reaches for it before the crosswalk tools.
 # OpenRouter executes it server-side (no local dispatch) and grounds results back.
 {"type":"openrouter:web_search",
   "parameters":{"engine":"auto","max_results":5,"search_context_size":"medium"}},
 {"type":"function","function":{"name":"crosswalk_string_search",
   "description":"Exact substring search over ASA Crosswalk SURGICAL descriptors. Pass 1-3 keyword terms (AND-matched). Each returned row gives the full crosswalk entry: surgical code+descriptor, mapped anesthesia code, base_units, alternate_anes_codes (other valid codes to consider for that procedure), and any coding comment/instruction text.",
   "parameters":{"type":"object","properties":{"terms":{"type":"array","items":{"type":"string"}}},"required":["terms"]}}},
 {"type":"function","function":{"name":"crosswalk_embedding_search",
   "description":"Semantic (meaning-based) search over ASA Crosswalk surgical descriptors. Pass a free-text procedure description; returns most-similar full rows (with anesthesia code, base_units, alternate_anes_codes, comment/instruction) + similarity score. Best for synonyms.",
   "parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}},
 {"type":"function","function":{"name":"crosswalk_cpt_lookup",
   "description":"Verify a candidate ANESTHESIA CPT code: returns its official descriptor, example surgical procedures that map to it, and any coding_notes (comments/instructions) attached to it. Use to confirm a candidate or to compare against an alternate code.",
   "parameters":{"type":"object","properties":{"anes_code":{"type":"string"}},"required":["anes_code"]}}},
]
DISPATCH={"crosswalk_string_search":lambda a:t_string_search(a["terms"]),
          "crosswalk_grep":lambda a:t_grep_search(a.get("pattern",""),whole_word=a.get("whole_word",False)),
          "crosswalk_embedding_search":lambda a:t_embedding_search(a["query"]),
          "crosswalk_cpt_lookup":lambda a:t_cpt_lookup(a["anes_code"]),
          "web_search":lambda a:t_web_search(a.get("query",""),a.get("num",5))}

AGENT_INSTRUCTIONS="""

==================== HOW TO WORK (AGENT MODE) ====================
In addition to the rules and code list above, you have these tools:
 - web_search: real-time web search. USE THIS FIRST to find the standard anesthesia (ASA) CPT code for the documented procedure.
 - crosswalk_string_search(terms): exact keyword substring search over the official 2025 ASA Crosswalk
 - crosswalk_embedding_search(query): semantic/meaning search over the crosswalk (handles synonyms)
 - crosswalk_cpt_lookup(anes_code): verify a candidate anesthesia code's meaning + example procedures

REQUIRED FIRST STEP: web_search the standard anesthesia CPT code for this exact procedure (e.g. "anesthesia CPT code for total knee arthroplasty", "ASA anesthesia code transurethral resection prostate"). Then corroborate with the crosswalk. Prefer a code both web AND crosswalk support.

PRECEDENCE (most important first):
 1. The CRITICAL CODING RULES above (e.g. colonoscopy 00811 vs 00812) ALWAYS win. Do NOT let the crosswalk override an explicit rule.
 2. The ADDITIONAL CUSTOM INSTRUCTIONS above (group-specific "If the Procedure is X then code is Y") take priority over the crosswalk. If the documented procedure matches one, use that code.
 3. Use the crosswalk tools to INVESTIGATE procedures NOT covered by the rules/custom instructions, or to verify a candidate.

WORK IN A LOOP: read procedure + diagnosis, apply rules/custom-instructions first, then use the crosswalk for anything uncertain. Verify candidates with crosswalk_cpt_lookup. Mind SITE, APPROACH (open/laparoscopic/percutaneous/transcatheter), DEPTH (subcutaneous/subfascial), VARIANT (simple/radical, diagnostic/therapeutic). Keep going until confident; do not answer prematurely.

USE THE FULL CROSSWALK ROW. Each result includes more than just a code:
 - alternate_codes: the official list of alternate anesthesia codes for that procedure, each with its descriptor.
 - comment / instruction / coding_notes: official ASA coding guidance.
 - base_units: relative magnitude of the procedure.

★★★ HARD RULE — ALTERNATE SELECTION (this is the #1 source of error, follow exactly) ★★★
When a crosswalk row lists alternate_codes, the primary code is NOT automatically the answer. The ASA rule is: "Selection of either the primary anesthesia code or of an alternate is DETERMINED BY THE SITE (and depth/approach/variant) of the surgical procedure." So you MUST:
 1. Read the EXACT body site, depth, and approach from the document (e.g. "breast", "lower leg", "upper abdomen", "instrumented multi-level spine", "total knee arthroplasty").
 2. Look at the primary code's descriptor AND every alternate's descriptor with crosswalk_cpt_lookup.
 3. Pick the ONE code whose descriptor matches THIS case's site/depth/variant — even if that means choosing an alternate over the primary, or a more specific code (e.g. breast 00402, not generic 00400; TURP 00914, not generic 00910; total knee 01402; instrumented spine 00670, not generic lumbar 00630; upper-abdomen 00790 vs lower-abdomen 00840 — by where the surgery actually is).
 4. NEVER default to the generic "not otherwise specified" code when a listed alternate names the specific site/procedure documented.
Explicitly state the site you read and which code's descriptor matches it before answering.

★★★ GENERAL RULE — DON'T BE SHY: COMPARE COMPETING CANDIDATES ★★★
Do NOT commit to the first plausible code you find. Before every final answer you MUST
have looked up the descriptors of AT LEAST TWO candidate codes and explicitly compared
them. A single body region can map to several different anesthesia codes that differ by
the TISSUE LAYER / STRUCTURE actually operated on — bone vs. integumentary/soft-tissue
(skin, subcutaneous, mass, cyst, lesion, node) vs. joint vs. nerve/muscle/tendon/fascia.
So read the procedure's OBJECT (what structure the surgeon acted on), not just its
location. Whenever the region could plausibly belong to more than one tissue layer, look
up a candidate from EACH plausible layer with crosswalk_cpt_lookup, then pick the one
whose descriptor names the actual structure documented and reject the others. State which
tissue layer the procedure targets and why the losing candidates were rejected.

★★★ GENERAL DISAMBIGUATION RULES ★★★
 - EYE (00140 vs 00142 vs 00145): 00140 = eye, not otherwise specified; 00142 = LENS surgery (cataract extraction, IOL, phacoemulsification); 00145 = VITREORETINAL surgery (vitrectomy, retinal detachment, scleral buckle). A cataract/lens case is 00142, NOT the generic 00140.
 - 01968 is an OB ADD-ON code (cesarean following neuraxial labor analgesia) that is only ever billed alongside a base code — it is NEVER a main-line / primary anesthesia code, so NEVER output 01968 as your answer.

You ALSO have web search. REQUIRED: before giving your final answer, run AT LEAST ONE web search to confirm the standard anesthesia CPT code for the specific procedure in this document (e.g. search "anesthesia CPT code for tympanoplasty" or "what ASA anesthesia code for transurethral resection of prostate"). This is mandatory whenever you are choosing between a generic "not otherwise specified" code and a more specific one, or whenever the crosswalk result is not an obvious exact match. Trust a code you can verify by web search over a crosswalk guess.

When done, STOP calling tools and reply with ONLY a JSON object:
{"code":"00860","confidence":"high|medium|low","procedure":"...","explanation":"one or two sentences"}"""

_SYS_CACHE={}
def build_system_prompt(custom_instructions=None):
    sys.path.insert(0,f"{ROOT}/backend"); sys.path.insert(0,f"{ROOT}/backend/general-coding")
    from predict_general import load_cpt_codes, _get_cpt_prompt
    base=_get_cpt_prompt(load_cpt_codes(),include_code_list=True)
    if custom_instructions and custom_instructions.strip():
        base+=f"\n\nADDITIONAL CUSTOM INSTRUCTIONS:\n{custom_instructions.strip()}"
    return base+AGENT_INSTRUCTIONS

def parse_final(text):
    import re
    if not text: return None
    obj=None
    m=re.search(r"\{.*\}",text,re.DOTALL)
    if m:
        try: obj=json.loads(m.group(0))
        except: obj=None
    if obj is None:
        # JSON malformed (model stutter, truncation, etc.) — fall back to first anesthesia code.
        # Prefer a code that appears after "code" if present, else the first 0XXXX in the text.
        cm=re.search(r'"code"\s*:\s*"?(0\d{4})', text) or re.search(r'\b(0\d{4})\b', text)
        if not cm: return None
        return {"predicted_cpt":cm.group(1), "explanation":text[:200], "_recovered":True}
    code=str(obj.get("code") or obj.get("predicted_cpt") or obj.get("cpt") or "").strip()
    if not code:
        cm=re.search(r'\b(0\d{4})\b', text)
        if cm: code=cm.group(1)
    if code.isdigit(): code=code.zfill(5)
    obj["predicted_cpt"]=code
    return obj

def pdf_images(pdf_path,n_pages):
    sys.path.insert(0,f"{ROOT}/backend/general-coding")
    from predict_general import pdf_pages_to_base64_images
    return pdf_pages_to_base64_images(pdf_path,n_pages=n_pages)

def forced_web_lookup(images, model):
    """Option A pre-step: force a web search for the standard anesthesia CPT of the
    documented procedure, BEFORE the crosswalk loop. Returns grounded text + #citations.
    gemini won't web-search on its own inside the main loop, so we do it deterministically."""
    url="https://openrouter.ai/api/v1/chat/completions"
    headers={"Authorization":f"Bearer {OR_KEY}","Content-Type":"application/json"}
    content=[{"type":"text","text":(
        "Look at the procedure and diagnosis in this anesthesia record. "
        "Search the web to determine the STANDARD anesthesia (ASA) CPT code billed for THIS exact procedure "
        "(consider site, approach, and whether a specific code applies vs a generic 'not otherwise specified' code). "
        "Answer in 2-3 sentences: name the procedure, then the anesthesia CPT code that web sources indicate, with brief justification.")}]
    for im in images: content.append({"type":"image_url","image_url":{"url":f"data:image/png;base64,{im}"}})
    payload={"model":model,"messages":[{"role":"user","content":content}],
             "tools":[{"type":"openrouter:web_search","parameters":{"engine":"auto","max_results":5}}],
             "usage":{"include":True}}
    last=None
    for attempt in range(5):
        try: r=requests.post(url,headers=headers,json=payload,timeout=240)
        except requests.exceptions.RequestException as e: last=e; time.sleep(min(2**attempt,20)); continue
        if r.status_code in (429,500,502,503,504): time.sleep(min(2**attempt,20)); continue
        r.raise_for_status()
        d=r.json(); msg=d["choices"][0]["message"]
        return {"text":(msg.get("content") or "").strip(),
                "citations":len(msg.get("annotations") or [])}
    if last: raise last
    return {"text":"", "citations":0}

# self-hosted vLLM (Qwen3.8-27B) via ngrok — OpenAI-compatible.
# VLLM_BASE is read from env because the free ngrok URL ROTATES on box restart;
# update the VLLM_BASE env var when it changes. If the vLLM call fails, the agent
# falls back to VLLM_FALLBACK_MODEL (gemini-3.7) so CPT never breaks.
VLLM_BASE=os.environ.get("VLLM_BASE","https://b296-2001-41d0-304-300-00-8823.ngrok-free.app/v1")
VLLM_KEY=os.environ.get("VLLM_KEY","sk-8pflTBC8CURdh9PVyLepI2ZQBvOc_oYroFm4js3du7I")
VLLM_MODELS={"Qwen/Qwen3.6-35B-A3B-FP8","nvidia/Qwen3.6-35B-A3B-NVFP4","unsloth/Qwen3.8-27B-NVFP4","vllm"}
VLLM_FALLBACK_MODEL=os.environ.get("VLLM_FALLBACK_MODEL","google/gemini-3.7-flash")
VLLM_THINKING=os.environ.get("VLLM_THINKING","0")=="1"   # toggle thinking via env
# Qwen3-VL processor rejects images larger than ~1250w/~1650h; resize before send.
VLLM_MAX_W, VLLM_MAX_LONG = 1200, 1536
_VLLM_SSL=ssl.create_default_context(); _VLLM_SSL.check_hostname=False; _VLLM_SSL.verify_mode=ssl.CERT_NONE
import urllib.request as _urlreq

def _is_vllm(model):
    return (model in VLLM_MODELS or model.startswith("Qwen/")
            or model.startswith("nvidia/") or model.startswith("unsloth/"))

# vLLM tool set — EXACTLY the validated local-harness set: real regex grep +
# semantic search + cpt_lookup + Serper web (vLLM can't use OpenRouter's server-side
# web_search, so we provide Serper as a function tool). Drops the substring
# crosswalk_string_search in favour of the more powerful crosswalk_grep.
_base_func_tools=[t for t in TOOLS_SPEC if t.get("type")=="function"
                  and t["function"]["name"]!="crosswalk_string_search"]
FUNC_TOOLS=[_GREP_TOOL_SPEC]+_base_func_tools+[_WEB_TOOL_SPEC]

def _resize_msg_images_for_vllm(messages):
    """Resize any base64 images in the message content to the Qwen3-VL safe
    envelope (<=1200x1536). Returns a NEW messages list; original untouched so a
    gemini fallback still sees full-res images."""
    try:
        import io as _io, base64 as _b64
        from PIL import Image as _Img
    except Exception:
        return messages
    out=[]
    for m in messages:
        c=m.get("content")
        if not isinstance(c,list):
            out.append(m); continue
        nc=[]
        for part in c:
            if isinstance(part,dict) and part.get("type")=="image_url":
                url=(part.get("image_url") or {}).get("url","")
                if url.startswith("data:image"):
                    try:
                        b64=url.split(",",1)[1]
                        im=_Img.open(_io.BytesIO(_b64.b64decode(b64)))
                        w,h=im.size; s=min(1.0,VLLM_MAX_W/w,VLLM_MAX_LONG/max(w,h))
                        if s<1.0:
                            im=im.resize((max(1,int(w*s)),max(1,int(h*s))),_Img.LANCZOS)
                        buf=_io.BytesIO(); im.convert("RGB").save(buf,"JPEG",quality=90)
                        nurl="data:image/jpeg;base64,"+_b64.b64encode(buf.getvalue()).decode()
                        nc.append({"type":"image_url","image_url":{"url":nurl}})
                        continue
                    except Exception:
                        pass
            nc.append(part)
        out.append({**m,"content":nc})
    return out

# Per-thread provenance: which model+provider ACTUALLY served the last call(s) in
# this thread's run_agent invocation. Lets the output record qwen-vllm vs a
# gemini fallback, rather than just the requested model.
import threading as _threading
_PROVENANCE=_threading.local()

def _prov_reset():
    _PROVENANCE.model=None; _PROVENANCE.provider=None; _PROVENANCE.fell_back=False
def _prov_set(model, provider, fell_back=False):
    _PROVENANCE.model=model; _PROVENANCE.provider=provider
    if fell_back: _PROVENANCE.fell_back=True
def get_last_provenance():
    """Return (model, provider, fell_back) actually used by the current thread's run."""
    return (getattr(_PROVENANCE,"model",None),
            getattr(_PROVENANCE,"provider",None),
            getattr(_PROVENANCE,"fell_back",False))

def call_openrouter(messages,model,tool_choice="auto"):
    if _is_vllm(model):
        try:
            r=_call_vllm(_resize_msg_images_for_vllm(messages),model,tool_choice)
            _prov_set(model, "vllm-ngrok")
            return r
        except Exception as e:
            # vLLM box down / ngrok URL rotated / any failure -> fall back to a
            # cloud model so CPT never breaks. Uses full-res original messages.
            print(f"    ⚠️  vLLM call failed ({str(e)[:80]}); falling back to {VLLM_FALLBACK_MODEL}")
            _prov_set(VLLM_FALLBACK_MODEL, "openrouter", fell_back=True)
            return call_openrouter(messages, VLLM_FALLBACK_MODEL, tool_choice)
    url="https://openrouter.ai/api/v1/chat/completions"
    headers={"Authorization":f"Bearer {OR_KEY}","Content-Type":"application/json"}
    payload={"model":model,"messages":messages,"tools":TOOLS_SPEC,"tool_choice":tool_choice,"usage":{"include":True}}
    last=None
    for attempt in range(6):
        try:
            r=requests.post(url,headers=headers,json=payload,timeout=240)
        except requests.exceptions.RequestException as e:
            last=e; time.sleep(min(2**attempt,20)); continue
        if r.status_code in (429,500,502,503,504):
            time.sleep(min(2**attempt,20)); continue
        r.raise_for_status()
        # provenance: if we didn't already mark a fallback, record this cloud model
        if not getattr(_PROVENANCE,"fell_back",False):
            _prov_set(model, "openrouter")
        return r.json()
    if last: raise last
    r.raise_for_status()

def _call_vllm(messages,model,tool_choice="auto"):
    # Send the requested model through as-is (the endpoint serves both
    # Qwen/Qwen3.6-35B-A3B-FP8 and nvidia/Qwen3.6-35B-A3B-NVFP4). Map the
    # generic "vllm" alias to the FP8 default.
    actual = "Qwen/Qwen3.6-35B-A3B-FP8" if model=="vllm" else model
    headers={"Authorization":f"Bearer {VLLM_KEY}","Content-Type":"application/json","ngrok-skip-browser-warning":"true"}
    payload={"model":actual,"messages":messages,"tools":FUNC_TOOLS,"tool_choice":tool_choice,
             "chat_template_kwargs":{"enable_thinking":VLLM_THINKING}}
    last=None
    # Fewer retries + shorter timeout so a down box fails over to gemini quickly.
    for attempt in range(3):
        try:
            req=_urlreq.Request(f"{VLLM_BASE}/chat/completions",
                data=json.dumps(payload).encode(),headers=headers)
            r=_urlreq.urlopen(req,timeout=180,context=_VLLM_SSL)
            return json.load(r)
        except Exception as e:
            last=e; time.sleep(min(2**attempt,8)); continue
    raise last

FORCE_CROSSWALK_MANDATE="""

==================== ★★★ MANDATORY CROSSWALK PROTOCOL ★★★ ====================
You are FORBIDDEN from answering from memory or from the code list alone. For EVERY case you MUST physically consult the ASA Crosswalk tools before producing any JSON answer. This is non-negotiable and applies even when you think you already know the code.

Required procedure, in order:
 1. Read the exact surgical procedure + site/approach/variant from the document.
 2. Call crosswalk_string_search with the key procedure terms (e.g. ["septoplasty"], ["turbinate"], ["knee","arthroscopy"]). If it returns nothing useful, call crosswalk_embedding_search with a full free-text description.
 3. Take the anesthesia code(s) the crosswalk returns and VERIFY the best candidate with crosswalk_cpt_lookup to confirm its descriptor matches this case's site/depth/variant.
 4. ONLY THEN output the final JSON, and your explanation MUST cite the specific crosswalk surgical row (surg code + descriptor) you relied on.

If you output a final answer without having called the crosswalk, that answer is INVALID. Do not skip the crosswalk because the procedure "seems obvious" — the obvious-looking cases (nose vs radical sinus, plain knee vs lower-leg, ear tube vs intraoral) are exactly where picking from memory goes wrong. Let the crosswalk decide the specificity.

★★★ HARD RULE — MULTIPLE PROCEDURES → HIGHER BASE UNITS WINS ★★★
When the document lists TWO OR MORE distinct surgical procedures done in the same session, you bill ONE anesthesia code: the code for the procedure with the MOST base units (the larger/primary procedure). You do NOT bill the smaller add-on procedure's code.
 - Every crosswalk search result includes a "base_units" field. Use it: look up each procedure, compare base_units, and choose the anesthesia code of the procedure with the HIGHER base_units.
 - Worked example (memorize this): a case with BOTH ear tube placement / tympanostomy AND adenoidectomy.
     · Tympanostomy (ventilating tube), surg 69436 → anesthesia 00126, base_units = 4
     · Adenoidectomy, surg 42830 → anesthesia 00170, base_units = 5
   00170 has more base units (5 > 4), so the answer is **00170**, NOT 00126. The ear-tube code 00126 is the smaller procedure and is dropped.
 - Generalize this to any pair: tube+adenoids, myringotomy+tonsils, etc. Search BOTH procedures in the crosswalk, compare base_units, and pick the higher one. Only fall back to a single code when the document truly documents a single procedure.
"""

def run_agent(pdf_path,model,n_pages=50,max_steps=14,verbose=True,custom_instructions=None,force_web=False,force_crosswalk=False):
    _prov_reset()
    key=(custom_instructions or "")+("|FCW" if force_crosswalk else "")
    if key not in _SYS_CACHE:
        sp=build_system_prompt(custom_instructions)
        if force_crosswalk: sp=sp+FORCE_CROSSWALK_MANDATE
        _SYS_CACHE[key]=sp
    system_prompt=_SYS_CACHE[key]
    imgs=pdf_images(pdf_path,n_pages)
    content=[{"type":"text","text":"Determine the single most appropriate anesthesia CPT code for this case."}]
    for im in imgs: content.append({"type":"image_url","image_url":{"url":f"data:image/png;base64,{im}"}})
    messages=[{"role":"system","content":system_prompt},{"role":"user","content":content}]
    trace=[]
    _xwalk_calls=[0]  # count of real crosswalk function calls so far
    # Option A: forced web-search pre-step. Inject grounded web evidence BEFORE the loop.
    if force_web:
        try:
            wl=forced_web_lookup(imgs, model)
            if wl["text"]:
                trace.append({"step":-1,"tool":"web_search","citations":wl["citations"],"forced":True})
                if verbose: print(f"[pre] WEB ({wl['citations']} cites): {wl['text'][:200]}")
                messages.append({"role":"user","content":
                    "WEB SEARCH EVIDENCE (from searching the web for this procedure's standard anesthesia CPT):\n"
                    + wl["text"] +
                    "\n\nWeigh this web evidence together with the crosswalk and the rules above. If the web evidence "
                    "names a specific anesthesia code that fits this procedure better than a generic one, prefer it. "
                    "You may still use the crosswalk tools to verify."})
        except Exception as e:
            if verbose: print(f"[pre] web lookup failed: {e}")
    for step in range(max_steps):
        last_step=step==max_steps-1
        if last_step:
            messages.append({"role":"user","content":"Stop investigating. Do NOT call tools. Output ONLY the final JSON now."})
        # force_crosswalk: on the very first step, REQUIRE a tool call (can't answer from memory)
        if force_crosswalk and step==0 and not last_step:
            tc="required"
        elif last_step:
            tc="none"
        else:
            tc="auto"
        resp=call_openrouter(messages,model,tool_choice=tc)
        if not isinstance(resp,dict) or "choices" not in resp:
            # OpenRouter returned an error body (e.g. payload too large, rate limit) — bail gracefully
            err=(resp.get("error") if isinstance(resp,dict) else None) or {"message":"no choices"}
            if verbose: print(f"[step {step}] API error: {json.dumps(err)[:200]}")
            return f'{{"code":"","error":{json.dumps(str(err))[:300]}}}', trace
        msg=resp["choices"][0]["message"]
        # detect web search via annotations (usage.web_search_requests is unreliable)
        wsr=(resp.get("usage") or {}).get("web_search_requests")
        n_ann=len(msg.get("annotations") or [])
        if wsr or n_ann: trace.append({"step":step,"tool":"web_search","requests":wsr,"citations":n_ann})
        messages.append(msg)
        calls=msg.get("tool_calls") or []
        if not calls:
            final=(msg.get("content") or "").strip()
            # force_crosswalk: refuse a final answer if the model never consulted the crosswalk
            if force_crosswalk and _xwalk_calls[0]==0 and not last_step:
                if verbose: print(f"[step {step}] REJECTED premature answer — no crosswalk call yet, pushing back")
                messages.append({"role":"user","content":
                    "You have NOT called the ASA Crosswalk yet. That is required before answering. "
                    "Call crosswalk_string_search (or crosswalk_embedding_search) for this procedure NOW, "
                    "then verify with crosswalk_cpt_lookup, before you give any JSON answer."})
                continue
            if verbose: print(f"\n[step {step}] FINAL: {final}")
            return final,trace
        for c in calls:
            name=(c.get("function") or {}).get("name","")
            try: args=json.loads((c.get("function") or {}).get("arguments") or "{}")
            except: args={}
            if name not in DISPATCH:
                # server-resolved tool (e.g. openrouter:web_search) or unknown — OpenRouter
                # handles it; just acknowledge so the conversation stays valid.
                if verbose: print(f"[step {step}] (server/unknown tool: {name or c.get('type')})")
                messages.append({"role":"tool","tool_call_id":c.get("id"),"name":name or "server_tool",
                                 "content":json.dumps({"note":"handled by server"})})
                continue
            if verbose: print(f"[step {step}] {name}({args})")
            try: result=DISPATCH[name](args)
            except Exception as e: result={"error":str(e)}
            if name.startswith("crosswalk_"): _xwalk_calls[0]+=1
            trace.append({"step":step,"tool":name,"args":args})
            if verbose: print(f"    -> {json.dumps(result)[:240]}")
            messages.append({"role":"tool","tool_call_id":c["id"],"name":name,"content":json.dumps(result)})
    return None,trace

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("pdf"); ap.add_argument("--model",default="google/gemini-3.5-flash"); ap.add_argument("--pages",type=int,default=2)
    a=ap.parse_args()
    print(f"=== CPT AGENT | model={a.model} | pages={a.pages} ===\nPDF: {os.path.basename(a.pdf)}\n")
    t0=time.time(); final,trace=run_agent(a.pdf,a.model,a.pages)
    print(f"\n=== {time.time()-t0:.0f}s, {len(trace)} tool calls ===\nFINAL: {final}")

---
name: Base prompts frontend tab
description: Added frontend tab for editing base CPT/ICD prompts and CPT codes list, stored in Railway PostgreSQL
type: project
---

Built a "Base Prompts & CPT Codes" tab under Settings in the Vue frontend (App.vue).

**Backend:**
- `base_prompts` table in Railway PostgreSQL (created by `init_database()` in db_utils.py)
- CRUD functions in `db_utils.py`: `get_all_base_prompts`, `get_base_prompt`, `upsert_base_prompt`, `delete_base_prompt`
- API endpoints in `main.py`: GET/PUT/DELETE `/api/base-prompts/{name}`, POST `/api/base-prompts/sync-from-files`
- Sync endpoint extracts prompts from `predict_general.py` and `cpt_codes.txt` into the database

**Frontend (App.vue):**
- Tab added to Settings dropdown as "📄 Base Prompts & CPT Codes"
- Data properties: `basePrompts`, `basePromptsLoading`, `editingBasePrompt`, etc. (around line 9921)
- Methods: `loadBasePrompts`, `saveBasePrompt`, `createBasePrompt`, `deleteBasePrompt`, `syncBasePromptsFromFiles`
- Tab HTML inserted before the Prediction Instructions tab (around line 9062)

**Current state:** Prompts are stored in DB and editable from UI, but `predict_general.py` still reads from hardcoded strings — NOT from the database at runtime. Editing in the UI updates the DB, but the Python file must also be manually updated to match. Runtime loading from DB is the next step.

**Why:** The base prompt ASA comorbidity rule was the main driver for this — different groups need different conventions (SIO wants no comorbidities for ASA 1-2, PCE-PMC wants them always). Having prompts in the DB allows per-deployment customization without code changes.

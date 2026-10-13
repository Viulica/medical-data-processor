<template>
  <div class="roster">
    <div class="roster-bar">
      <div class="roster-pills">
        <span class="pill"><b>{{ countBy('CRNA') }}</b> CRNA</span>
        <span class="pill"><b>{{ countBy('MD') }}</b> MD</span>
        <span v-if="otherCredCount" class="pill"><b>{{ otherCredCount }}</b> other</span>
        <span v-if="issueCount" class="pill pill-bad"><b>{{ issueCount }}</b> to fix</span>
        <span v-else-if="rows.length" class="pill pill-ok">All codes valid</span>
      </div>
      <div class="roster-actions">
        <button type="button" class="r-btn" @click="toggleText">{{ textMode ? 'Back to table' : 'Edit as text' }}</button>
        <button v-if="!textMode" type="button" class="r-btn r-btn-dark" @click="addRow">+ Add provider</button>
      </div>
    </div>

    <!-- Text mode: paste the Provider Mapping tool output, as before -->
    <div v-if="textMode" class="roster-text">
      <textarea
        v-model="draftText"
        class="form-textarea roster-textarea"
        rows="10"
        spellcheck="false"
        placeholder="Paste output from the Provider Mapping tool here...
Billable CRNA providers:
Smith, John, CRNA (MedNet Code: 7)

Billable MD providers:
Johnson, Robert, MD (MedNet Code: 1)"
      ></textarea>
      <p class="roster-hint">Switch back to the table to check it. Lines without "(MedNet Code: …)" are kept but can't be matched to pasted codes.</p>
    </div>

    <template v-else>
      <input
        v-if="rows.length > 8"
        v-model="query"
        type="search"
        class="roster-search"
        placeholder="Search name or MedNet code"
        aria-label="Search providers"
      />

      <div v-if="!rows.length && !extras.length" class="roster-empty">
        No providers yet. Add them one by one, or use “Edit as text” to paste the Provider Mapping tool output.
      </div>

      <div v-else class="roster-table-wrap">
        <table class="roster-table">
          <thead>
            <tr><th class="c-code">MedNet code</th><th>Last name</th><th>First name</th><th class="c-cred">Role</th><th class="c-del"></th></tr>
          </thead>
          <tbody>
            <tr v-for="row in visibleRows" :key="row.id" :class="{ 'row-new': row.isNew }">
              <td class="c-code">
                <input v-model="row.code" class="cell mono" :class="{ bad: issues[row.id] && issues[row.id].code }"
                       :aria-label="'MedNet code'" @input="onEdit" />
                <span v-if="issues[row.id] && issues[row.id].code" class="cell-err">{{ issues[row.id].code }}</span>
              </td>
              <td>
                <input v-model="row.last" class="cell" :class="{ bad: issues[row.id] && issues[row.id].last }"
                       aria-label="Last name" @input="onEdit" />
                <span v-if="issues[row.id] && issues[row.id].last" class="cell-err">{{ issues[row.id].last }}</span>
              </td>
              <td><input v-model="row.first" class="cell" aria-label="First name and middle initial" @input="onEdit" /></td>
              <td class="c-cred">
                <select v-model="row.cred" class="cell" aria-label="Role" @change="onEdit">
                  <option v-for="c in CREDS" :key="c" :value="c">{{ c }}</option>
                  <option v-if="!CREDS.includes(row.cred)" :value="row.cred">{{ row.cred || '—' }}</option>
                </select>
              </td>
              <td class="c-del"><button type="button" class="del" :aria-label="'Remove ' + row.last" @click="removeRow(row)">✕</button></td>
            </tr>
          </tbody>
        </table>
      </div>

      <div v-if="extras.length" class="roster-extras">
        <div class="extras-head">
          <b>{{ extras.length }} line{{ extras.length === 1 ? '' : 's' }} without a MedNet code</b>
          <span>Kept as written and shown to the AI, but never matched to a pasted code.</span>
        </div>
        <div v-for="x in extras" :key="x.id" class="extra-line">
          <code>{{ x.text }}</code>
          <button type="button" class="del" aria-label="Remove line" @click="removeExtra(x)">✕</button>
        </div>
      </div>
    </template>
  </div>
</template>

<script>
// Structured editor for a template's provider_mapping text. Reads and writes the exact text format the backend
// parses (provider_annotation_utils.parse_provider_mapping), so nothing changes server-side:
//   Billable CRNA providers:
//   LAST, FIRST M, CRNA (MedNet Code: 7)
// It flags what the parser would silently get wrong: a code used twice (the last one wins), missing codes or names.
const CODE_RE = /\(MedNet Code:\s*([A-Za-z0-9]+)\)/i;
const CREDS = ['CRNA', 'MD'];
let uid = 0;

function sectionOf(line) {
  const u = line.toUpperCase();
  if (u.includes('CRNA') && u.includes('PROVIDER')) return 'CRNA';
  if (u.includes('MD') && u.includes('PROVIDER')) return 'MD';
  return null;
}

function parse(text) {
  const rows = [], extras = [];
  let section = '';
  for (const raw of (text || '').split('\n')) {
    const line = raw.trim();
    if (!line) continue;
    const sec = sectionOf(line);
    if (sec && !CODE_RE.test(line)) { section = sec; continue; }
    const m = line.match(CODE_RE);
    if (!m) { extras.push({ id: ++uid, text: line, section }); continue; }
    const parts = line.slice(0, m.index).split(',').map((p) => p.trim()).filter(Boolean);
    let cred = section || 'CRNA';
    if (parts.length > 1 && /^(CRNA|MD|DO|SRNA|CAA|RESIDENT)$/i.test(parts[parts.length - 1])) cred = parts.pop().toUpperCase();
    rows.push({ id: ++uid, code: m[1].toUpperCase(), last: (parts[0] || '').toUpperCase(),
                first: parts.slice(1).join(' ').replace(/\s+/g, ' ').toUpperCase(), cred, section: section || cred });
  }
  return { rows, extras };
}

function serialize(rows, extras) {
  const line = (r) => `${r.last.trim()}, ${r.first.trim()}${r.first.trim() ? ', ' : ''}${r.cred} (MedNet Code: ${r.code.trim()})`
    .replace(/^, /, '');
  const groups = [['CRNA', 'Billable CRNA providers:'], ['MD', 'Billable MD providers:']];
  const out = [];
  const loose = extras.filter((x) => !x.section).map((x) => x.text);
  if (loose.length) out.push(loose.join('\n'));
  for (const [cred, header] of groups) {
    const body = [
      ...extras.filter((x) => x.section === cred).map((x) => x.text),
      ...rows.filter((r) => (CREDS.includes(r.cred) ? r.cred : r.section) === cred).map(line),
    ];
    if (body.length) out.push([header, ...body].join('\n'));
  }
  const others = rows.filter((r) => !CREDS.includes(r.cred) && !CREDS.includes(r.section)).map(line);
  if (others.length) out.push(['Billable other providers:', ...others].join('\n'));
  return out.join('\n\n');
}

export default {
  name: 'ProviderRosterEditor',
  props: { modelValue: { type: String, default: '' } },
  emits: ['update:modelValue'],
  data() {
    const { rows, extras } = parse(this.modelValue);
    return { CREDS, rows, extras, query: '', textMode: false, draftText: this.modelValue || '', lastEmitted: this.modelValue || '' };
  },
  computed: {
    issues() {
      const out = {}, byCode = {};
      for (const r of this.rows) {
        const e = {};
        const code = r.code.trim().toUpperCase();
        if (!code) e.code = 'Code needed';
        else if (!/^[A-Z0-9]+$/.test(code)) e.code = 'Letters and numbers only';
        else (byCode[code] = byCode[code] || []).push(r.id);
        if (!r.last.trim()) e.last = 'Last name needed';
        out[r.id] = e;
      }
      for (const ids of Object.values(byCode)) {
        if (ids.length > 1) ids.forEach((id) => { out[id].code = `Code used ${ids.length}×, only the last row would be matched`; });
      }
      return out;
    },
    issueCount() { return Object.values(this.issues).filter((e) => Object.keys(e).length).length; },
    otherCredCount() { return this.rows.filter((r) => !CREDS.includes(r.cred)).length; },
    visibleRows() {
      const q = this.query.trim().toUpperCase();
      if (!q) return this.rows;
      return this.rows.filter((r) => `${r.code} ${r.last} ${r.first}`.toUpperCase().includes(q));
    },
  },
  watch: {
    // A different template was opened (or the parent replaced the text): reload.
    modelValue(v) {
      if (v === this.lastEmitted) return;
      const { rows, extras } = parse(v);
      this.rows = rows; this.extras = extras; this.draftText = v || ''; this.lastEmitted = v || '';
    },
    draftText(v) { if (this.textMode) this.emit(v); },
  },
  methods: {
    countBy(cred) { return this.rows.filter((r) => r.cred === cred).length; },
    emit(text) { this.lastEmitted = text; this.$emit('update:modelValue', text); },
    onEdit() { this.emit(serialize(this.rows, this.extras)); },
    addRow() {
      this.query = '';
      this.rows.unshift({ id: ++uid, code: '', last: '', first: '', cred: 'CRNA', section: 'CRNA', isNew: true });
      this.$nextTick(() => { const el = this.$el.querySelector('tbody input'); if (el) el.focus(); });
    },
    removeRow(row) { this.rows = this.rows.filter((r) => r !== row); this.onEdit(); },
    removeExtra(x) { this.extras = this.extras.filter((e) => e !== x); this.onEdit(); },
    toggleText() {
      if (this.textMode) {
        const { rows, extras } = parse(this.draftText);
        this.rows = rows; this.extras = extras; this.textMode = false;
      } else {
        this.draftText = this.lastEmitted || serialize(this.rows, this.extras);
        this.textMode = true;
      }
    },
  },
};
</script>

<style scoped>
.roster { display: grid; gap: 10px; }
.roster-bar { display: flex; flex-wrap: wrap; gap: 8px; justify-content: space-between; align-items: center; }
.roster-pills, .roster-actions { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
.pill { font-size: 12px; padding: 3px 9px; border-radius: 999px; background: #f1f5f9; color: #475569; }
.pill b { color: #0f172a; font-variant-numeric: tabular-nums; }
.pill-bad { background: #fef2f2; color: #b91c1c; border: 1px solid #fca5a5; }
.pill-bad b { color: #b91c1c; }
.pill-ok { background: #f0fdf4; color: #15803d; border: 1px solid #86efac; }
.r-btn { padding: 5px 10px; border: 1px solid #cbd5e1; background: white; color: #0f172a; border-radius: 6px; font-size: 12.5px; font-weight: 500; cursor: pointer; }
.r-btn:hover { border-color: #94a3b8; }
.r-btn-dark { background: #0f172a; color: white; border-color: #0f172a; }
.r-btn-dark:hover { background: #1e293b; }
.roster-search { width: 100%; padding: 6px 10px; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 13px; }
.roster-empty { padding: 14px; border: 1px dashed #cbd5e1; border-radius: 8px; color: #64748b; font-size: 13px; }
.roster-table-wrap { max-height: 340px; overflow: auto; border: 1px solid #e2e8f0; border-radius: 8px; }
.roster-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.roster-table th { position: sticky; top: 0; z-index: 1; background: #f8fafc; text-align: left; font-size: 11px; font-weight: 600;
  letter-spacing: .04em; text-transform: uppercase; color: #64748b; padding: 7px 6px; border-bottom: 1px solid #e2e8f0; }
.roster-table td { padding: 2px 4px; border-bottom: 1px solid #f1f5f9; vertical-align: top; }
.roster-table tr:last-child td { border-bottom: 0; }
.row-new td { background: #f0fdf4; }
.c-code { width: 92px; }
.c-cred { width: 78px; }
.c-del { width: 28px; text-align: center; }
.cell { width: 100%; padding: 5px 6px; border: 1px solid transparent; border-radius: 4px; background: transparent; font-size: 13px; color: #0f172a; }
.cell:hover { border-color: #e2e8f0; }
.cell:focus { border-color: #3b82f6; outline: none; background: white; }
.cell.bad { border-color: #f87171; background: #fef2f2; }
.cell.mono, .extra-line code { font-family: 'SF Mono', Menlo, Consolas, monospace; }
.cell-err { display: block; color: #b91c1c; font-size: 11px; line-height: 1.3; padding: 0 6px 4px; }
.del { border: 0; background: none; color: #94a3b8; cursor: pointer; padding: 6px 4px; font-size: 12px; }
.del:hover { color: #b91c1c; }
.roster-extras { border: 1px solid #fde68a; background: #fffbeb; border-radius: 8px; padding: 8px 10px; display: grid; gap: 4px; }
.extras-head { display: grid; font-size: 12.5px; color: #92400e; }
.extras-head span { color: #a16207; font-size: 12px; }
.extra-line { display: flex; justify-content: space-between; align-items: center; gap: 8px; }
.extra-line code { font-size: 12px; color: #713f12; word-break: break-word; }
.roster-textarea { font-family: 'SF Mono', Menlo, Consolas, monospace; font-size: 12.5px; }
.roster-hint { font-size: 0.85em; color: #666; margin: 4px 0 0; }
</style>

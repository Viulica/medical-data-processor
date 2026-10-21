<template>
  <div class="tr-overlay" @keydown="onKey">
    <div class="tr-shell">
      <!-- Left: the chart -->
      <section class="tr-pdf">
        <div class="tr-pdf-bar">
          <span class="tr-file" :title="cur && cur.source_file">{{ cur ? cur.source_file : '' }}</span>
          <div class="tr-pages" v-if="pageCount > 1">
            <button
              v-for="p in pageCount" :key="p" type="button"
              class="tr-page" :class="{ on: p === page, hint: p === hintPage }"
              :title="p === hintPage ? 'The page the AI read the times from' : `Page ${p}`"
              @click="goPage(p)"
            >{{ p }}</button>
          </div>
        </div>
        <div v-if="pdfError" class="tr-empty">{{ pdfError }}</div>
        <iframe v-else-if="pdfSrc" :key="pdfSrc" :src="pdfSrc" title="Patient chart"></iframe>
        <div v-else class="tr-empty">Loading chart…</div>
      </section>

      <!-- Right: the three values -->
      <aside class="tr-side">
        <header class="tr-head">
          <div>
            <h2>DOS &amp; Times</h2>
            <p class="tr-sub" v-if="data">{{ data.group || '—' }} · batch {{ data.batch || '—' }}</p>
          </div>
          <button type="button" class="tr-x" title="Close (Esc)" @click="close">✕</button>
        </header>

        <div v-if="loading" class="tr-empty">Loading…</div>
        <div v-else-if="loadError" class="tr-empty">{{ loadError }}</div>
        <template v-else-if="cur">
          <div class="tr-progress">
            <div class="tr-bar"><div class="tr-fill" :style="{ width: pct + '%' }"></div></div>
            <span>{{ doneCount }} / {{ visible.length }} checked</span>
          </div>
          <label class="tr-toggle">
            <input type="checkbox" v-model="onlyFlagged" @change="onFilterChange" />
            Only flagged ({{ flaggedCount }})
          </label>

          <div class="tr-patient">
            <div class="tr-pos">{{ pos + 1 }} of {{ visible.length }}</div>
            <strong>{{ patientName }}</strong>
            <div class="tr-meta">
              <span v-if="cur['Case #']">Case {{ cur['Case #'] }}</span>
              <span v-if="cur['Patient DOB']">DOB {{ cur['Patient DOB'] }}</span>
            </div>
            <div class="tr-flags" v-if="cur._needs_review.length">
              <span v-for="f in cur._needs_review" :key="f" class="tr-flag">{{ flagLabel(f) }}</span>
            </div>
          </div>

          <form class="tr-form" @submit.prevent="confirmNext">
            <label class="tr-field">
              <span>Date of service</span>
              <input
                ref="dos" v-model="draft.dos" class="tr-in" :class="{ bad: !dosOk, changed: changed.dos }"
                inputmode="numeric" placeholder="MM/DD/YYYY" autocomplete="off"
                @focus="$event.target.select()" @blur="draft.dos = normDate(draft.dos)"
              />
            </label>
            <div class="tr-times">
              <label class="tr-field">
                <span>An Start</span>
                <input
                  ref="start" v-model="draft.start" class="tr-in tr-time" :class="{ bad: !startOk, changed: changed.start }"
                  inputmode="numeric" placeholder="HHMM" maxlength="5" autocomplete="off"
                  @focus="$event.target.select()" @input="draft.start = liveTime(draft.start)" @blur="draft.start = normTime(draft.start)"
                />
              </label>
              <label class="tr-field">
                <span>An Stop</span>
                <input
                  ref="stop" v-model="draft.stop" class="tr-in tr-time" :class="{ bad: !stopOk, changed: changed.stop }"
                  inputmode="numeric" placeholder="HHMM" maxlength="5" autocomplete="off"
                  @focus="$event.target.select()" @input="draft.stop = liveTime(draft.stop)" @blur="draft.stop = normTime(draft.stop)"
                />
              </label>
            </div>
            <div class="tr-dur" :class="{ warn: durWarn }">{{ durText }}</div>

            <div class="tr-nav">
              <button type="button" class="tr-btn" :disabled="pos === 0" @click="move(-1)">← Back</button>
              <button type="submit" class="tr-btn tr-primary" :disabled="!valid">
                {{ isDirty ? 'Save & next' : 'OK & next' }} ↵
              </button>
            </div>
            <p v-if="saveError" class="tr-err">{{ saveError }}</p>
          </form>

          <div class="tr-keys">
            <div><kbd>Enter</kbd> save &amp; next · <kbd>Tab</kbd> next box</div>
            <div><kbd>Alt</kbd>+<kbd>←</kbd>/<kbd>→</kbd> back / skip · <kbd>Alt</kbd>+<kbd>1</kbd>–<kbd>9</kbd> page</div>
            <div>Type <code>1251</code> for 12:51. A stop before the start is read as the next day.</div>
          </div>

          <div v-if="allDone" class="tr-done">
            All {{ visible.length }} checked.
            <button type="button" class="tr-btn tr-primary" @click="download('csv')">Download CSV</button>
            <button type="button" class="tr-btn" @click="download('xlsx')">XLSX</button>
          </div>
        </template>
        <div v-else-if="data" class="tr-empty">No charts to review{{ onlyFlagged ? ' (none flagged)' : '' }}.</div>
      </aside>
    </div>
  </div>
</template>

<script>
import axios from "axios";

const FLAG_LABELS = {
  "missing-time": "time missing",
  "no-dos": "no date of service",
  "start-date-not-dos": "start date ≠ DOS",
  "stop-before-start": "stop before start",
  "over-8h": "over 8 hours",
  "under-10min": "under 10 minutes",
};

// "10/19/2026 09:37", "9/30/2026 9:22:00 AM", "10/19/2026"
const DT_RE = /(\d{1,2})\/(\d{1,2})\/(\d{4})(?:\s+(\d{1,2}):(\d{2})(?::(\d{2}))?\s*([AaPp][Mm])?)?/;

function pad(n) {
  return String(n).padStart(2, "0");
}

function parseDT(v) {
  const m = DT_RE.exec(String(v || ""));
  if (!m) return null;
  let h = m[4] != null ? Number(m[4]) : null;
  const ap = (m[7] || "").toUpperCase();
  if (h != null && ap === "PM" && h < 12) h += 12;
  if (h != null && ap === "AM" && h === 12) h = 0;
  return {
    date: `${pad(m[1])}/${pad(m[2])}/${m[3]}`,
    time: h != null ? `${pad(h)}:${m[5]}` : "",
    ampm: !!m[7],
    seconds: m[6] != null,
    padded: m[1].length === 2,
  };
}

function minutesOf(t) {
  const m = /^(\d{2}):(\d{2})$/.exec(t || "");
  return m ? Number(m[1]) * 60 + Number(m[2]) : null;
}

function addDay(date) {
  const [mo, d, y] = date.split("/").map(Number);
  const x = new Date(y, mo - 1, d + 1);
  return `${pad(x.getMonth() + 1)}/${pad(x.getDate())}/${x.getFullYear()}`;
}

// Write date + 24h time in the style the row already uses (24h for BAP; AM/PM with seconds elsewhere).
function compose(original, date, time) {
  const o = parseDT(original) || { ampm: false, seconds: false, padded: true };
  let [mo, d, y] = date.split("/");
  if (!o.padded) {
    mo = String(Number(mo));
    d = String(Number(d));
  }
  let [h, mi] = time.split(":").map(Number);
  if (!o.ampm) return `${mo}/${d}/${y} ${pad(h)}:${pad(mi)}${o.seconds ? ":00" : ""}`;
  const ap = h >= 12 ? "PM" : "AM";
  h = h % 12 || 12;
  return `${mo}/${d}/${y} ${h}:${pad(mi)}:00 ${ap}`;
}

export default {
  name: "TimesReview",
  props: {
    jobId: { type: String, required: true },
    backendUrl: { type: String, required: true },
  },
  emits: ["close", "saved"],
  data() {
    return {
      data: null,
      loading: true,
      loadError: "",
      onlyFlagged: false,
      pos: 0,
      draft: { dos: "", start: "", stop: "" },
      done: {}, // row -> true once confirmed
      pdfs: {}, // row -> { url, pages }
      pdfError: "",
      page: 1,
      saving: false,
      saveError: "",
    };
  },
  computed: {
    visible() {
      if (!this.data) return [];
      return this.onlyFlagged ? this.data.rows.filter((r) => r._needs_review.length) : this.data.rows;
    },
    cur() {
      return this.visible[this.pos] || null;
    },
    flaggedCount() {
      return this.data ? this.data.rows.filter((r) => r._needs_review.length).length : 0;
    },
    doneCount() {
      return this.visible.filter((r) => this.done[r._row]).length;
    },
    allDone() {
      return this.visible.length > 0 && this.doneCount === this.visible.length;
    },
    pct() {
      return this.visible.length ? Math.round((100 * this.doneCount) / this.visible.length) : 0;
    },
    patientName() {
      const r = this.cur || {};
      return [r["Patient Last Name"], r["Patient First Name"]].filter(Boolean).join(", ") || "(no name)";
    },
    hintPage() {
      const n = parseInt(this.cur && this.cur["Time Page"], 10);
      return Number.isFinite(n) && n > 0 ? n : null;
    },
    pageCount() {
      const p = this.cur && this.pdfs[this.cur._row];
      return p ? p.pages : 0;
    },
    pdfSrc() {
      const p = this.cur && this.pdfs[this.cur._row];
      return p ? `${p.url}#page=${this.page}&zoom=page-width` : "";
    },
    original() {
      const r = this.cur || {};
      const s = parseDT(r["An Start"]);
      const e = parseDT(r["An Stop"]);
      const dos = parseDT(r["Charge Date"]) || s;
      return { dos: dos ? dos.date : "", start: s ? s.time : "", stop: e ? e.time : "" };
    },
    changed() {
      return {
        dos: this.draft.dos !== this.original.dos,
        start: this.draft.start !== this.original.start,
        stop: this.draft.stop !== this.original.stop,
      };
    },
    isDirty() {
      return this.changed.dos || this.changed.start || this.changed.stop;
    },
    dosOk() {
      return /^\d{2}\/\d{2}\/\d{4}$/.test(this.draft.dos);
    },
    startOk() {
      return this.draft.start === "" || minutesOf(this.draft.start) != null;
    },
    stopOk() {
      return this.draft.stop === "" || minutesOf(this.draft.stop) != null;
    },
    valid() {
      return this.dosOk && this.startOk && this.stopOk && !this.saving;
    },
    duration() {
      const a = minutesOf(this.draft.start);
      const b = minutesOf(this.draft.stop);
      if (a == null || b == null) return null;
      return { min: b > a ? b - a : b + 1440 - a, overnight: b <= a };
    },
    durText() {
      const d = this.duration;
      if (!d) return "Duration: —";
      return `Duration: ${Math.floor(d.min / 60)}h ${pad(d.min % 60)}m${d.overnight ? " · stop is next day" : ""}`;
    },
    durWarn() {
      const d = this.duration;
      return !!d && (d.overnight || d.min > 480 || d.min < 10);
    },
  },
  watch: {
    cur(row) {
      if (!row) return;
      this.draft = { ...this.original };
      this.saveError = "";
      this.page = this.hintPage || 1;
      this.loadPdf(row);
      this.prefetch();
      this.$nextTick(() => this.focusFirst());
    },
  },
  async mounted() {
    try {
      this.done = JSON.parse(localStorage.getItem(this.storeKey()) || "{}");
    } catch (e) {
      this.done = {};
    }
    try {
      const r = await axios.get(`${this.backendUrl}/api/unified-results/${this.jobId}/review`, { params: { profile: "times" } });
      this.data = r.data;
      const first = this.visible.findIndex((x) => !this.done[x._row]);
      this.pos = first >= 0 ? first : 0;
    } catch (e) {
      this.loadError = `Failed to load: ${e.response?.data?.detail || e.message}`;
    } finally {
      this.loading = false;
    }
  },
  beforeUnmount() {
    Object.values(this.pdfs).forEach((p) => p.url && URL.revokeObjectURL(p.url));
  },
  methods: {
    flagLabel(f) {
      return FLAG_LABELS[f] || f;
    },
    storeKey() {
      return `times-review:${this.jobId}`;
    },
    remember() {
      try {
        localStorage.setItem(this.storeKey(), JSON.stringify(this.done));
      } catch (e) {
        /* private window: progress just isn't remembered */
      }
    },
    focusFirst() {
      // the box usually still has focus from the last chart, so select explicitly: typing then replaces the value
      const el = this.$refs.start;
      if (el) {
        el.focus();
        el.select();
      }
    },
    // "1251" -> "12:51" while typing
    liveTime(v) {
      const d = String(v || "").replace(/\D/g, "").slice(0, 4);
      return d.length > 2 ? `${d.slice(0, 2)}:${d.slice(2)}` : d;
    },
    normTime(v) {
      const d = String(v || "").replace(/\D/g, "");
      if (!d) return "";
      const s = d.length <= 2 ? d.padStart(2, "0") + "00" : d.padStart(4, "0").slice(0, 4);
      const h = Number(s.slice(0, 2));
      const m = Number(s.slice(2));
      return h < 24 && m < 60 ? `${s.slice(0, 2)}:${s.slice(2)}` : v;
    },
    normDate(v) {
      const s = String(v || "").trim();
      let m = /^(\d{1,2})\D(\d{1,2})\D(\d{2}|\d{4})$/.exec(s);
      if (!m && /^\d{8}$/.test(s)) m = [s, s.slice(0, 2), s.slice(2, 4), s.slice(4)];
      if (!m) return s;
      const y = m[3].length === 2 ? `20${m[3]}` : m[3];
      return `${pad(m[1])}/${pad(m[2])}/${y}`;
    },
    async loadPdf(row) {
      this.pdfError = "";
      if (this.pdfs[row._row]) return;
      try {
        await this.fetchPdf(row);
      } catch (e) {
        if (this.cur && this.cur._row === row._row) this.pdfError = `Could not load the chart: ${e.response?.status === 404 ? "not in the input ZIP" : e.message}`;
      }
    },
    async fetchPdf(row) {
      if (this.pdfs[row._row] || !row.source_file) return;
      const r = await axios.get(
        `${this.backendUrl}/api/unified-results/${this.jobId}/pdf/${encodeURIComponent(row.source_file)}`,
        { responseType: "blob" },
      );
      const text = await r.data.text();
      // page count from the PDF itself (page objects, else the page tree's /Count); 0 = unknown, buttons hidden
      const counts = [...text.matchAll(/\/Count\s+(\d+)/g)].map((m) => Number(m[1]));
      const pages = Math.max((text.match(/\/Type\s*\/Page(?![s\w])/g) || []).length, ...counts, 0);
      this.pdfs = { ...this.pdfs, [row._row]: { url: URL.createObjectURL(r.data), pages } };
    },
    prefetch() {
      for (const r of this.visible.slice(this.pos + 1, this.pos + 4)) this.fetchPdf(r).catch(() => {});
    },
    goPage(p) {
      if (p >= 1 && (!this.pageCount || p <= this.pageCount)) this.page = p;
    },
    move(step) {
      const n = this.pos + step;
      if (n >= 0 && n < this.visible.length) this.pos = n;
    },
    onFilterChange() {
      const first = this.visible.findIndex((x) => !this.done[x._row]);
      this.pos = first >= 0 ? first : 0;
    },
    async confirmNext() {
      this.draft.dos = this.normDate(this.draft.dos);
      this.draft.start = this.normTime(this.draft.start);
      this.draft.stop = this.normTime(this.draft.stop);
      if (!this.valid) return;
      const row = this.cur;
      if (this.isDirty && !(await this.save(row))) return;
      this.done = { ...this.done, [row._row]: true };
      this.remember();
      // next unchecked chart after this one, else just the next one
      const after = this.visible.findIndex((x, i) => i > this.pos && !this.done[x._row]);
      if (after >= 0) this.pos = after;
      else if (this.pos < this.visible.length - 1) this.pos += 1;
    },
    async save(row) {
      const { dos, start, stop } = this.draft;
      const a = minutesOf(start);
      const b = minutesOf(stop);
      const stopDate = a != null && b != null && b <= a ? addDay(dos) : dos;
      const values = {
        "Charge Date": compose(row["Charge Date"], dos, "00:00").split(" ")[0],
        "An Start": start ? compose(row["An Start"] || row["An Stop"], dos, start) : "",
        "An Stop": stop ? compose(row["An Stop"] || row["An Start"], stopDate, stop) : "",
      };
      const edits = Object.entries(values)
        .filter(([f, v]) => (this.data.editable_fields || []).includes(f) && v !== String(row[f] ?? ""))
        .map(([field, value]) => ({ row: row._row, field, value }));
      if (!edits.length) return true;
      this.saving = true;
      this.saveError = "";
      try {
        await axios.put(`${this.backendUrl}/api/unified-results/${this.jobId}/review`, { edits }, { params: { profile: "times" } });
        edits.forEach((e) => (row[e.field] = e.value));
        this.$emit("saved", edits.length);
        return true;
      } catch (e) {
        this.saveError = `Save failed: ${e.response?.data?.detail || e.message}`;
        return false;
      } finally {
        this.saving = false;
      }
    },
    onKey(e) {
      if (e.key === "Escape") {
        e.preventDefault();
        this.close();
      } else if (e.altKey && e.key === "ArrowRight") {
        e.preventDefault();
        this.move(1);
      } else if (e.altKey && e.key === "ArrowLeft") {
        e.preventDefault();
        this.move(-1);
      } else if (e.altKey && /^[1-9]$/.test(e.key)) {
        e.preventDefault();
        this.goPage(Number(e.key));
      }
    },
    download(format) {
      window.location.href = `${this.backendUrl}/api/unified-results/${this.jobId}/export?format=${format}`;
    },
    close() {
      this.$emit("close");
    },
  },
};
</script>

<style scoped>
.tr-overlay { position: fixed; inset: 0; z-index: 1100; background: rgba(15, 23, 42, 0.6); display: flex; padding: 0.75rem; }
.tr-shell { flex: 1; display: flex; background: #f8fafc; border-radius: 12px; overflow: hidden; box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3); }
.tr-pdf { flex: 1; display: flex; flex-direction: column; min-width: 0; background: #525659; }
.tr-pdf-bar { display: flex; justify-content: space-between; align-items: center; gap: 0.75rem; padding: 0.4rem 0.75rem; background: white; border-bottom: 1px solid #e5e7eb; font-size: 0.8rem; color: #334155; }
.tr-file { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.tr-pages { display: flex; gap: 0.25rem; flex-shrink: 0; }
.tr-page { min-width: 1.9rem; padding: 0.2rem 0.4rem; border: 1px solid #cbd5e1; border-radius: 6px; background: white; cursor: pointer; font-size: 0.8rem; }
.tr-page.hint { border-color: #f59e0b; }
.tr-page.on { background: #1e293b; color: white; border-color: #1e293b; }
.tr-pdf iframe { flex: 1; border: 0; width: 100%; }
.tr-side { width: 360px; flex-shrink: 0; display: flex; flex-direction: column; gap: 0.9rem; padding: 1rem 1.1rem; overflow-y: auto; border-left: 1px solid #e5e7eb; }
.tr-head { display: flex; justify-content: space-between; align-items: flex-start; }
.tr-head h2 { margin: 0; font-size: 1.15rem; }
.tr-sub { margin: 0.15rem 0 0; color: #64748b; font-size: 0.85rem; }
.tr-x { border: 0; background: transparent; font-size: 1.1rem; cursor: pointer; color: #475569; }
.tr-progress { display: flex; align-items: center; gap: 0.6rem; font-size: 0.8rem; color: #475569; }
.tr-bar { flex: 1; height: 6px; background: #e2e8f0; border-radius: 3px; overflow: hidden; }
.tr-fill { height: 100%; background: #16a34a; transition: width 0.2s; }
.tr-toggle { font-size: 0.85rem; color: #334155; display: flex; gap: 0.4rem; align-items: center; }
.tr-patient { background: white; border: 1px solid #e5e7eb; border-radius: 10px; padding: 0.7rem 0.8rem; }
.tr-pos { font-size: 0.75rem; color: #64748b; margin-bottom: 0.2rem; }
.tr-meta { display: flex; gap: 0.8rem; font-size: 0.8rem; color: #475569; margin-top: 0.2rem; flex-wrap: wrap; }
.tr-flags { display: flex; gap: 0.3rem; flex-wrap: wrap; margin-top: 0.5rem; }
.tr-flag { font-size: 0.72rem; padding: 0.1rem 0.45rem; border-radius: 999px; background: #fef3c7; color: #92400e; }
.tr-form { display: flex; flex-direction: column; gap: 0.7rem; }
.tr-field { display: flex; flex-direction: column; gap: 0.25rem; font-size: 0.8rem; color: #475569; }
.tr-times { display: grid; grid-template-columns: 1fr 1fr; gap: 0.7rem; }
.tr-in { font: 600 1.4rem/1.2 ui-monospace, SFMono-Regular, Menlo, monospace; padding: 0.45rem 0.6rem; border: 2px solid #cbd5e1; border-radius: 8px; background: white; color: #0f172a; width: 100%; box-sizing: border-box; }
.tr-time { font-size: 1.9rem; text-align: center; letter-spacing: 0.04em; }
.tr-in:focus { outline: none; border-color: #2563eb; box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.2); }
.tr-in.changed { border-color: #7c3aed; background: #faf5ff; }
.tr-in.bad { border-color: #dc2626; background: #fef2f2; }
.tr-dur { font-size: 0.85rem; color: #475569; }
.tr-dur.warn { color: #b45309; font-weight: 600; }
.tr-nav { display: flex; gap: 0.6rem; }
.tr-btn { padding: 0.6rem 0.9rem; border: 1px solid #cbd5e1; border-radius: 8px; background: white; cursor: pointer; font-size: 0.9rem; }
.tr-btn:disabled { opacity: 0.5; cursor: default; }
.tr-primary { flex: 1; background: #1e293b; color: white; border-color: #1e293b; font-weight: 600; }
.tr-err { color: #dc2626; font-size: 0.85rem; margin: 0; }
.tr-keys { font-size: 0.75rem; color: #64748b; display: flex; flex-direction: column; gap: 0.3rem; border-top: 1px solid #e5e7eb; padding-top: 0.7rem; }
.tr-keys kbd { font: 0.7rem ui-monospace, monospace; border: 1px solid #cbd5e1; border-bottom-width: 2px; border-radius: 4px; padding: 0 0.3rem; background: white; }
.tr-done { display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap; background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 10px; padding: 0.7rem; font-size: 0.85rem; color: #166534; }
.tr-empty { padding: 2rem; text-align: center; color: #94a3b8; }
@media (max-width: 900px) {
  .tr-shell { flex-direction: column; }
  .tr-pdf { min-height: 55vh; }
  .tr-side { width: auto; border-left: 0; border-top: 1px solid #e5e7eb; }
}
</style>

// solokit frontend
// Vanilla ESM. Talks to the FastAPI backend at the same origin.

const API = "";  // same origin

// --- status indicator -------------------------------------------------------

const statusDot = document.getElementById("status-dot");
const statusText = document.getElementById("status-text");

async function checkHealth() {
    try {
        const r = await fetch(`${API}/healthz`);
        if (!r.ok) throw new Error(`status ${r.status}`);
        const data = await r.json();
        statusDot.className = "dot ok";
        statusText.textContent = `v${data.version} · ${data.corpora.join(", ")}`;
    } catch (err) {
        statusDot.className = "dot error";
        statusText.textContent = "server unreachable";
    }
}
checkHealth();

// --- toast ------------------------------------------------------------------

const toastEl = document.getElementById("toast");
let toastTimer = null;

function toast(msg, kind = "ok") {
    toastEl.textContent = msg;
    toastEl.className = `toast ${kind}`;
    toastEl.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { toastEl.hidden = true; }, 3000);
}

// --- pattern search ---------------------------------------------------------

const searchForm = document.getElementById("search-form");
const patternInput = document.getElementById("pattern");
const transformationSelect = document.getElementById("transformation");
const limitInput = document.getElementById("limit");
const minSimInput = document.getElementById("min-similarity");
const minSimValue = document.getElementById("min-similarity-value");
const maxLenInput = document.getElementById("max-length-diff");
const maxLenValue = document.getElementById("max-length-diff-value");
const corporaGroup = document.getElementById("corpora-group");
const searchBtn = document.getElementById("search-btn");
const exampleBtn = document.getElementById("example-btn");
const searchResults = document.getElementById("search-results");
const resultsTbody = document.getElementById("results-tbody");
const resultsEmpty = document.getElementById("results-empty");
const resultsCount = document.getElementById("results-count");
const resultsTook = document.getElementById("results-took");

minSimInput.addEventListener("input", () => { minSimValue.textContent = minSimInput.value; });
maxLenInput.addEventListener("input", () => { maxLenValue.textContent = maxLenInput.value; });

exampleBtn.addEventListener("click", () => {
    // Bebop classic: descending 5th with chromatic neighbor — a Charlie Parker staple
    patternInput.value = "-1 -1 4 -5 -2";
    transformationSelect.value = "interval";
    toast("Loaded example pattern. Hit Search to run it.", "ok");
});

function parsePattern(raw) {
    return raw
        .trim()
        .split(/[\s,]+/)
        .filter(Boolean)
        .map(Number);
}

function getSelectedCorpora() {
    return Array.from(corporaGroup.querySelectorAll('input[type="checkbox"]:checked'))
        .map(cb => cb.value);
}

function setLoading(btn, loading, label = "Search") {
    btn.disabled = loading;
    const spinner = btn.querySelector(".spinner");
    spinner.hidden = !loading;
    const lbl = btn.querySelector(".btn-label");
    lbl.textContent = loading ? `${label}ing…` : label;
}

searchForm.addEventListener("submit", async (e) => {
    e.preventDefault();

    const pattern = parsePattern(patternInput.value);
    if (pattern.length < 2) {
        toast("Pattern needs at least 2 numbers.", "error");
        return;
    }
    if (pattern.some(n => Number.isNaN(n))) {
        toast("Pattern contains non-numeric values.", "error");
        return;
    }

    const corpora = getSelectedCorpora();
    if (corpora.length === 0) {
        toast("Pick at least one corpus.", "error");
        return;
    }

    const body = {
        pattern,
        transformation: transformationSelect.value,
        databases: corpora,
        min_similarity: parseFloat(minSimInput.value),
        max_length_difference: parseInt(maxLenInput.value, 10),
        limit: parseInt(limitInput.value, 10),
    };

    setLoading(searchBtn, true);
    searchResults.hidden = true;
    resultsTbody.innerHTML = "";

    try {
        const r = await fetch(`${API}/search`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });
        if (!r.ok) {
            const err = await r.json().catch(() => ({ detail: r.statusText }));
            throw new Error(err.detail || `HTTP ${r.status}`);
        }
        const data = await r.json();
        renderResults(data);
        if (data.errors && Object.keys(data.errors).length > 0) {
            const failed = Object.keys(data.errors).join(", ");
            if (data.matches.length > 0) {
                toast(`Partial results — ${failed} failed. See console for details.`, "error");
                console.warn("Search errors:", data.errors);
            } else {
                toast(`All requested corpora failed: ${failed}`, "error");
                console.error("Search errors:", data.errors);
            }
        } else if (data.matches.length === 0) {
            toast("No matches found. Try lowering min similarity.", "ok");
        } else {
            toast(`${data.matches.length} match(es) in ${data.took_ms.toFixed(0)}ms`, "ok");
        }
    } catch (err) {
        toast(`Search failed: ${err.message}`, "error");
        console.error(err);
    } finally {
        setLoading(searchBtn, false);
    }
});

function renderResults(data) {
    searchResults.hidden = false;
    resultsCount.textContent = data.matches.length === 0 ? "" : `(${data.matches.length})`;
    resultsTook.textContent = `${data.took_ms.toFixed(0)}ms`;

    if (data.matches.length === 0) {
        resultsTbody.innerHTML = "";
        resultsEmpty.hidden = false;
        return;
    }
    resultsEmpty.hidden = true;

    const rows = data.matches.map(m => {
        const simPct = (m.similarity * 100).toFixed(0);
        return `
            <tr>
                <td>${simPct}%</td>
                <td>${m.edit_distance}</td>
                <td>${escapeHtml(m.performer)}</td>
                <td>${escapeHtml(m.title)}</td>
                <td>${m.database}</td>
                <td>${m.year ?? "—"}</td>
            </tr>`;
    }).join("");
    resultsTbody.innerHTML = rows;
}

function escapeHtml(s) {
    return String(s ?? "").replace(/[&<>"']/g, c => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
    }[c]));
}

// --- audio upload + transcribe ---------------------------------------------

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("audio-file");
const filenameEl = document.getElementById("audio-filename");
const transcribeBtn = document.getElementById("transcribe-btn");
const audioForm = document.getElementById("audio-form");
const transcriptionPanel = document.getElementById("transcription");
const notesPreview = document.getElementById("notes-preview");
const transcriptionMeta = document.getElementById("transcription-meta");
const derivedPattern = document.getElementById("derived-pattern");
const useDerivedBtn = document.getElementById("use-derived-btn");

let selectedFile = null;
let lastNotes = [];

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
        handleFile(e.dataTransfer.files[0]);
    }
});
fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) handleFile(e.target.files[0]);
});

function handleFile(file) {
    selectedFile = file;
    filenameEl.textContent = file.name;
    transcribeBtn.disabled = false;
}

audioForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!selectedFile) {
        toast("Choose a file first.", "error");
        return;
    }

    setLoading(transcribeBtn, true, "Transcribe");
    transcriptionPanel.hidden = true;
    notesPreview.innerHTML = "";

    try {
        const formData = new FormData();
        formData.append("file", selectedFile);
        const r = await fetch(`${API}/transcribe`, { method: "POST", body: formData });
        if (!r.ok) {
            const err = await r.json().catch(() => ({ detail: r.statusText }));
            throw new Error(err.detail || `HTTP ${r.status}`);
        }
        const data = await r.json();
        renderTranscription(data);
        toast(`Transcribed ${data.notes.length} notes.`, "ok");
    } catch (err) {
        toast(`Transcribe failed: ${err.message}`, "error");
        console.error(err);
    } finally {
        setLoading(transcribeBtn, false, "Transcribe");
    }
});

function renderTranscription(data) {
    lastNotes = data.notes;
    transcriptionPanel.hidden = false;
    transcriptionMeta.textContent =
        `${data.notes.length} notes · ${data.tempo_bpm.toFixed(0)} BPM · ${data.time_signature.join("/")}` +
        (data.key_signature ? ` · ${data.key_signature}` : "");

    // Show notes as colored chips
    notesPreview.innerHTML = data.notes.map(n => {
        const name = n.pitch != null ? midiToNoteName(n.pitch) : "—";
        return `<span class="note" title="MIDI ${n.pitch} · onset ${n.onset_beat.toFixed(2)} beats · dur ${n.duration_beats.toFixed(2)} beats">${name}</span>`;
    }).join(" ");

    // Derive interval pattern from the notes
    const intervals = deriveIntervals(data.notes);
    const patternStr = intervals.join(" ");
    derivedPattern.textContent = patternStr || "(need ≥ 2 notes with detected pitches)";

    useDerivedBtn.disabled = intervals.length < 2;
}

function deriveIntervals(notes) {
    // Take notes with detected pitches, compute consecutive intervals
    const pitches = notes.filter(n => n.pitch != null).map(n => n.pitch);
    if (pitches.length < 2) return [];
    const intervals = [];
    for (let i = 1; i < pitches.length; i++) {
        intervals.push(pitches[i] - pitches[i - 1]);
    }
    return intervals;
}

useDerivedBtn.addEventListener("click", () => {
    if (lastNotes.length < 2) return;
    const intervals = deriveIntervals(lastNotes);
    if (intervals.length < 2) {
        toast("Not enough pitched notes to derive a pattern.", "error");
        return;
    }
    patternInput.value = intervals.join(" ");
    transformationSelect.value = "interval";
    // Make sure WJAZD is checked (it's local and instant)
    const wjazzdCb = corporaGroup.querySelector('input[value="wjazzd"]');
    if (wjazzdCb && !wjazzdCb.checked) wjazzdCb.checked = true;
    // Switch to the search panel
    document.getElementById("search-panel").scrollIntoView({ behavior: "smooth", block: "start" });
    patternInput.focus();
    toast("Pattern loaded. Hit Search to run it.", "ok");
});

function midiToNoteName(midi) {
    const names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
    const oct = Math.floor(midi / 12) - 1;
    return `${names[midi % 12]}${oct}`;
}

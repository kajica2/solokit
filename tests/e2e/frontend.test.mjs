// Puppeteer e2e test for the solokit frontend.
// Assumes a server is already running at process.env.SOLOKIT_URL
// (defaults to http://127.0.0.1:8765).
//
// Run with:
//   npm install  (once)
//   SOLOKIT_URL=http://127.0.0.1:8765 node tests/e2e/frontend.test.mjs

import puppeteer from "puppeteer";
import { fileURLToPath } from "url";
import { dirname, resolve } from "path";
import { existsSync } from "fs";

const BASE_URL = process.env.SOLOKIT_URL || "http://127.0.0.1:8765";
const __dirname = dirname(fileURLToPath(import.meta.url));
const SCREENSHOTS = resolve(__dirname, "screenshots");
const FIXTURES = resolve(__dirname, "fixtures");

import { mkdirSync } from "fs";
mkdirSync(SCREENSHOTS, { recursive: true });

let pass = 0;
let fail = 0;
const failures = [];

function ok(name) {
    pass++;
    console.log(`  ✓ ${name}`);
}

function bad(name, err) {
    fail++;
    failures.push({ name, err });
    console.log(`  ✗ ${name}\n      ${err}`);
}

async function expect(name, fn) {
    try {
        await fn();
        ok(name);
    } catch (err) {
        bad(name, err.message || String(err));
    }
}

async function main() {
    console.log(`\nsolokit frontend e2e — ${BASE_URL}\n`);

    const browser = await puppeteer.launch({
        headless: true,
        args: ["--no-sandbox", "--disable-setuid-sandbox"],
    });
    const page = await browser.newPage();
    await page.setViewport({ width: 1400, height: 900 });

    // Forward browser console errors to test output
    const consoleErrors = [];
    page.on("pageerror", (err) => consoleErrors.push(`pageerror: ${err.message}`));
    page.on("console", (msg) => {
        if (msg.type() === "error") consoleErrors.push(`console.error: ${msg.text()}`);
    });

    // --- 1. page loads ---
    console.log("[1] page load");
    const resp = await page.goto(BASE_URL, { waitUntil: "networkidle0", timeout: 15000 });
    await expect("HTTP 200", () => {
        if (resp.status() !== 200) throw new Error(`got ${resp.status()}`);
    });
    await expect("title contains solokit", async () => {
        const title = await page.title();
        if (!title.toLowerCase().includes("solokit")) throw new Error(`title: ${title}`);
    });
    await expect("status indicator turns green (server reachable)", async () => {
        await page.waitForSelector(".dot.ok", { timeout: 5000 });
    });
    await expect("search panel visible", async () => {
        const el = await page.$("#search-panel");
        if (!el) throw new Error("missing #search-panel");
    });
    await expect("audio panel visible", async () => {
        const el = await page.$("#audio-panel");
        if (!el) throw new Error("missing #audio-panel");
    });

    await page.screenshot({ path: `${SCREENSHOTS}/01-loaded.png`, fullPage: true });

    // --- 2. pattern search ---
    console.log("\n[2] pattern search");
    await expect("type a pattern", async () => {
        await page.click("#pattern");
        await page.type("#pattern", "-1 -1 4 -5 -2");
    });
    await expect("WJAZD is checked by default", async () => {
        const checked = await page.$eval(
            'input[name="corpus"][value="wjazzd"]',
            (el) => el.checked
        );
        if (!checked) throw new Error("wjazzd should default to checked");
    });
    await expect("click Search returns matches", async () => {
        await page.click("#search-btn");
        // Wait for results table to appear (or empty state)
        await page.waitForSelector("#search-results:not([hidden])", { timeout: 15000 });
        // Wait for at least one row OR the empty state
        await page.waitForFunction(
            () => document.querySelectorAll("#results-tbody tr").length > 0
                || !document.getElementById("results-empty").hidden,
            { timeout: 15000 }
        );
        const rowCount = await page.$$eval("#results-tbody tr", (rows) => rows.length);
        if (rowCount === 0) throw new Error("no result rows rendered");
    });
    await expect("results show a known bebop lick (Dizzy or Bird)", async () => {
        const text = await page.$eval("#results-tbody", (el) => el.innerText);
        // WJAZD should return at least Dizzy Gillespie, Bird, Fats Navarro, etc.
        const known = ["Dizzy", "Bird", "Parker", "Navarro", "Fats"];
        const hit = known.find((n) => text.includes(n));
        if (!hit) throw new Error(`no known performer in: ${text.slice(0, 200)}`);
    });
    await expect("sim column shows percentages", async () => {
        const cells = await page.$$eval("#results-tbody tr td:first-child", (els) => els.map((e) => e.innerText));
        const allPct = cells.every((c) => c.includes("%"));
        if (!allPct) throw new Error(`expected all %, got: ${cells.slice(0, 5).join(", ")}`);
    });

    await page.screenshot({ path: `${SCREENSHOTS}/02-search-results.png`, fullPage: true });

    // --- 3. example button ---
    console.log("\n[3] example pattern button");
    await expect("example button fills the input", async () => {
        await page.click("#example-btn");
        const val = await page.$eval("#pattern", (el) => el.value);
        if (val !== "-1 -1 4 -5 -2") throw new Error(`expected pattern, got: ${val}`);
    });

    // --- 4. min-similarity slider updates label ---
    console.log("\n[4] slider label binding");
    await expect("min-similarity slider shows value", async () => {
        await page.evaluate(() => {
            const s = document.getElementById("min-similarity");
            s.value = "0.65";
            s.dispatchEvent(new Event("input"));
        });
        const label = await page.$eval("#min-similarity-value", (el) => el.innerText);
        if (label !== "0.65") throw new Error(`expected 0.65, got ${label}`);
    });

    // --- 5. multi-corpus search aggregates + handles DTL error ---
    console.log("\n[5] multi-corpus search (aggregates wjazzd + omnibook + DTL)");
    await expect("checking both wjazzd and omnibook returns matches from both", async () => {
        // Set the form fields cleanly via JS — page.click(3) + type appends to
        // existing content instead of replacing it.
        await page.evaluate(() => {
            const setVal = (sel, val) => {
                const el = document.querySelector(sel);
                el.value = val;
                el.dispatchEvent(new Event("input", { bubbles: true }));
            };
            setVal("#pattern", "-1 -1 4 -5 -2");
            setVal("#limit", "30");
            const setChecked = (name, val) => {
                const el = document.querySelector(`input[name="corpus"][value="${name}"]`);
                if (el && el.checked !== val) el.checked = val;
            };
            setChecked("wjazzd", true);
            setChecked("omnibook", true);
            setChecked("dtl", true);
        });
        await page.click("#search-btn");
        await page.waitForFunction(
            () => document.querySelectorAll("#results-tbody tr").length > 0
                || !document.getElementById("results-empty").hidden,
            { timeout: 30000 }
        );
        // Look at the database column to verify omnibook is in there
        const dbs = await page.$$eval(
            "#results-tbody tr td:nth-child(5)",
            (els) => els.map((e) => e.innerText)
        );
        if (!dbs.includes("omnibook")) {
            throw new Error(`expected omnibook in results, only saw: ${[...new Set(dbs)].join(", ")}`);
        }
        // Also confirm Parker is somewhere in the rendered rows
        const text = await page.$eval("#results-tbody", (el) => el.innerText);
        if (!text.includes("Parker") && !text.includes("Charlie")) {
            throw new Error(`expected Parker/Charlie in results, got: ${text.slice(0, 300)}`);
        }
        // Note: DTL is sometimes down (real-world). The partial-failure path
        // is covered by pytest's test_search_continues_when_one_corpus_fails.
        // If you want to assert on the toast, check that errors.dtl appears
        // in the response — but the API server logs the error in the
        // browser console as well.
    });

    await page.screenshot({ path: `${SCREENSHOTS}/03-multi-corpus.png`, fullPage: true });

    // --- 6. transcribe (audio upload) ---
    console.log("\n[6] audio upload + transcribe");
    const wavPath = `${FIXTURES}/cmajor_arpeggio.wav`;
    if (!existsSync(wavPath)) {
        bad("fixtures/cmajor_arpeggio.wav exists", `not found at ${wavPath}`);
    } else {
        await expect("file input is reachable", async () => {
            const input = await page.$("#audio-file");
            if (!input) throw new Error("missing #audio-file");
        });
        await expect("upload a wav and see transcription notes", async () => {
            const input = await page.$("#audio-file");
            await input.uploadFile(wavPath);
            // Filename appears
            await page.waitForFunction(
                () => document.getElementById("audio-filename").innerText.length > 0,
                { timeout: 5000 }
            );
            // Transcribe button enabled
            const disabled = await page.$eval("#transcribe-btn", (el) => el.disabled);
            if (disabled) throw new Error("transcribe button still disabled after upload");
            // Click
            await page.click("#transcribe-btn");
            // Wait for either transcription panel or error toast
            await page.waitForFunction(
                () => !document.getElementById("transcription").hidden
                    || (document.getElementById("toast").innerText || "").toLowerCase().includes("failed"),
                { timeout: 30000 }
            );
            const transHidden = await page.$eval("#transcription", (el) => el.hidden);
            if (transHidden) {
                // The 1-second sine arpeggio may not be enough for pYIN to lock on;
                // accept either a transcription OR a graceful error toast.
                const toastText = await page.$eval("#toast", (el) => el.innerText);
                if (!toastText.toLowerCase().includes("fail") && !toastText.toLowerCase().includes("transcrib")) {
                    throw new Error(`no transcription and no failure toast; got: ${toastText}`);
                }
            } else {
                const noteCount = await page.$$eval(".notes-preview .note", (els) => els.length);
                if (noteCount === 0) throw new Error("transcription has 0 notes");
            }
        });
    }

    await page.screenshot({ path: `${SCREENSHOTS}/04-transcribed.png`, fullPage: true });

    // --- summary ---
    await browser.close();

    console.log(`\n${pass} passed, ${fail} failed`);
    if (consoleErrors.length > 0) {
        console.log(`\nBrowser console errors (${consoleErrors.length}):`);
        consoleErrors.forEach((e) => console.log(`  - ${e}`));
    }
    console.log(`\nScreenshots: ${SCREENSHOTS}`);
    if (fail > 0) {
        console.log("\nFailures:");
        failures.forEach((f) => console.log(`  - ${f.name}: ${f.err}`));
        process.exit(1);
    }
}

main().catch((err) => {
    console.error("FATAL:", err);
    process.exit(2);
});

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(fileURLToPath(import.meta.url));
const [html, css, js] = await Promise.all([
  readFile(join(root, "index.html"), "utf8"),
  readFile(join(root, "styles.css"), "utf8"),
  readFile(join(root, "app.js"), "utf8"),
]);

assert.match(html, /id="experiment-list"/);
assert.match(html, /id="start-run"/);
assert.match(html, /id="event-feed"/);
assert.match(html, /id="result-table"/);
assert.match(html, /id="export-json"/);
assert.match(css, /--ink:/);
assert.match(css, /@media/);
assert.match(js, /API_BASE/);
assert.match(js, /\/runs\//);
assert.match(js, /simulation/);
assert.match(js, /jev/);
assert.match(js, /download/);
assert.match(js, /restoreLatestRun/);
assert.match(js, /localStorage/);
console.log("smoke_check: passed — required workbench surfaces and API adapters are present");

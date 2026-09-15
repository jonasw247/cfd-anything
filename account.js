// CFD Anything - accounts and uploads, on Supabase (auth, the table and storage bucket "uploads"; set up with
// website/backend/schema.sql). Sign-in stays out of sight until "upload your own": the visitor picks a file first (it
// stays in the browser), signs in to send it (Google / GitHub / Microsoft, or a 6-digit code by e-mail) and finds the
// flow under their name. OAuth leaves the page, so the picked file waits in IndexedDB and is sent when the visitor comes
// back signed in. Signed in, a profile button sits top left: e-mail, the sent shapes and their status, sign out.
// Loaded before app.js, so the ?code= of the OAuth return is read before app.js rewrites the URL.
// SUPABASE_URL empty = not connected: the window says so and nothing is sent.

const SUPABASE_URL = "";                     // https://<project>.supabase.co (EU region)
const SUPABASE_ANON_KEY = "";                // the public "anon" key; row-level security in schema.sql guards the data
const PROVIDERS = ["google", "github"];      // OAuth providers switched on in the Supabase dashboard (also "azure")
const MAX_MB = 200;                          // = file_size_limit of the bucket
const FORMATS = [".stl", ".obj", ".ply", ".step", ".stp"];
// On-prem inference service (the worker behind the VPN, reached by a Cloudflare Tunnel): the file goes straight to it,
// it computes the flow and streams the result back, no sign-in. "" = off (same-origin service on localhost, for dev).
// the on-prem inference service (Cloudflare Tunnel); on localhost the local preview server (website/local/serve.py)
// answers on the same origin, so the page talks to that instead
const API_BASE = ["localhost", "127.0.0.1"].includes(location.hostname) ? "" : "https://api.cfd-anything.com";
const api = (path) => (API_BASE ? `${API_BASE.replace(/\/$/, "")}/${path}` : path);

const $ = (sel) => document.querySelector(sel);
const track = (name) => { try { window.goatcounter?.count?.({ path: name, title: name, event: true }); } catch { /* ignore */ } };

const LABEL = { google: "Google", github: "GitHub", azure: "Microsoft" };
const ICON = {
  google: '<svg viewBox="0 0 48 48" aria-hidden="true"><path fill="#EA4335" d="M24 9.5c3.5 0 6.7 1.2 9.2 3.6l6.9-6.9C35.9 2.4 30.5 0 24 0 14.6 0 6.6 5.4 2.6 13.3l8 6.2C12.5 13.7 17.8 9.5 24 9.5z"/><path fill="#4285F4" d="M46.1 24.5c0-1.6-.1-3.1-.4-4.5H24v9h12.4c-.5 2.9-2.2 5.3-4.6 7l7.4 5.7c4.3-4 6.9-9.9 6.9-17.2z"/><path fill="#FBBC05" d="M10.5 28.6c-.5-1.4-.8-3-.8-4.6s.3-3.2.8-4.6l-8-6.2C.9 16.5 0 20.1 0 24s.9 7.5 2.6 10.8l7.9-6.2z"/><path fill="#34A853" d="M24 48c6.5 0 11.9-2.1 15.9-5.8l-7.4-5.7c-2.1 1.4-4.8 2.3-8.5 2.3-6.2 0-11.5-4.2-13.4-9.9l-8 6.2C6.6 42.6 14.6 48 24 48z"/></svg>',
  github: '<svg viewBox="0 0 16 16" aria-hidden="true"><path fill="currentColor" d="M8 0a8 8 0 0 0-2.53 15.59c.4.07.55-.17.55-.38v-1.33c-2.23.48-2.7-1.07-2.7-1.07-.36-.92-.89-1.17-.89-1.17-.73-.5.05-.49.05-.49.8.06 1.23.83 1.23.83.72 1.22 1.87.87 2.33.66.07-.52.28-.87.5-1.07-1.78-.2-3.65-.89-3.65-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.6 7.6 0 0 1 4 0c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.28.82 2.15 0 3.07-1.87 3.75-3.66 3.95.29.25.54.73.54 1.48v2.2c0 .21.15.46.55.38A8 8 0 0 0 8 0z"/></svg>',
  azure: '<svg viewBox="0 0 22 22" aria-hidden="true"><path fill="#F25022" d="M0 0h10.5v10.5H0z"/><path fill="#7FBA00" d="M11.5 0H22v10.5H11.5z"/><path fill="#00A4EF" d="M0 11.5h10.5V22H0z"/><path fill="#FFB900" d="M11.5 11.5H22V22H11.5z"/></svg>',
};
const STATUS = { queued: "in the queue", running: "computing", done: "ready", failed: "could not be computed" };

const sb = SUPABASE_URL && window.supabase
  ? window.supabase.createClient(SUPABASE_URL, SUPABASE_ANON_KEY, { auth: { flowType: "pkce", detectSessionInUrl: false } })
  : null;
let session = null;
let pending = null;                           // { file }: picked, not sent yet
// The inference service (API_BASE, the on-prem worker) computes the flow with no sign-in and app.js adds the shape to
// the list; on localhost a same-origin service is used for development. Probed once; the upload path appears only if it
// answers, so the published site stays "under development" until API_BASE is set.
let local = false;
if (API_BASE || ["localhost", "127.0.0.1"].includes(location.hostname)) {
  fetch(api("api/health"), { cache: "no-store" }).then((r) => { local = r.ok; if (local && pending) showPicked(); }, () => {});
}

// ------------------------------------------------------------------------------------------------ picked file across the OAuth redirect
function idb(mode, op) {
  return new Promise((resolve, reject) => {
    const open = indexedDB.open("cfd-anything", 1);
    open.onupgradeneeded = () => open.result.createObjectStore("pending");
    open.onerror = () => reject(open.error);
    open.onsuccess = () => {
      const tx = open.result.transaction("pending", mode), req = op(tx.objectStore("pending"));
      tx.oncomplete = () => { open.result.close(); resolve(req.result); };
      tx.onerror = () => reject(tx.error);
    };
  });
}
const stash = (p) => idb("readwrite", (s) => s.put(p, "upload")).catch(() => {});
const unstash = () => idb("readonly", (s) => s.get("upload")).catch(() => undefined);
const dropStash = () => idb("readwrite", (s) => s.delete("upload")).catch(() => {});

// ------------------------------------------------------------------------------------------------ upload window
const dialog = $("#upload-dialog");
const steps = dialog.querySelectorAll("[data-step]");
const note = (id, text) => { $(id).textContent = text; };

function show(step) {
  for (const s of steps) s.hidden = s.dataset.step !== step;
}

function openUpload() {
  track("upload/open");
  note("#pick-note", "");
  note("#signin-note", "");
  showPicked();
  show("pick");
  if (!dialog.open) dialog.showModal();
}

function showPicked() {
  const f = pending?.file;
  $("#drop").classList.toggle("picked", !!f);
  $("#drop-main").innerHTML = f ? "" : "drop a file here or <u>choose one</u>";
  if (f) $("#drop-main").textContent = f.name;
  const size = f && (f.size < 1e5 ? `${Math.ceil(f.size / 1e3)} kB` : `${(f.size / 1e6).toFixed(1)} MB`);
  note("#drop-sub", f ? `${size} · choose another` : `${local ? "STL, OBJ or PLY" : "STL, OBJ, PLY or STEP"}, up to ${MAX_MB} MB`);
  $("#upload-next").disabled = !f;
  $("#upload-next").textContent = local ? "compute" : session ? "send" : "continue";
}

// Sends in the background: the window closes at once and app.js lists the shape with a ring right away
// (cfd:upload-start), then swaps in the service's answer (cfd:upload, replaces) or marks it failed (cfd:upload-failed).
async function computeLocal() {
  const { file } = pending, temp = `sending-${Date.now()}`;
  pending = null;
  $("#upload-file").value = "";                             // the same file can be picked again (else no change event)
  showPicked();
  note("#pick-note", "");
  dialog.close();
  window.dispatchEvent(new CustomEvent("cfd:upload-start", { detail: { name: temp, label: file.name.replace(/\.[^.]+$/, "") || "your shape" } }));
  let res;
  try {
    const r = await fetch(api(`api/upload?name=${encodeURIComponent(file.name)}`), { method: "POST", body: file });
    res = await r.json().catch(() => ({ error: `HTTP ${r.status}` }));
    if (!r.ok) throw new Error(res.detail || res.error || `HTTP ${r.status}`);
  } catch (err) {
    return window.dispatchEvent(new CustomEvent("cfd:upload-failed", { detail: { name: temp, error: err.message } }));
  }
  window.dispatchEvent(new CustomEvent("cfd:upload", { detail: { ...res, replaces: temp } }));
}

function pick(file) {
  if (!file) return;
  const ext = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
  if (!FORMATS.includes(ext)) return note("#pick-note", "Please pick an STL, OBJ, PLY or STEP file.");
  if (!file.size || file.size > MAX_MB * 1e6) return note("#pick-note", `The file has to be smaller than ${MAX_MB} MB.`);
  track("upload/file");
  pending = { file };
  note("#pick-note", "");
  showPicked();
}

$("#upload-file").addEventListener("change", (ev) => pick(ev.target.files[0]));
const drop = $("#drop");
drop.addEventListener("dragover", (ev) => { ev.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (ev) => { ev.preventDefault(); drop.classList.remove("over"); pick(ev.dataTransfer.files[0]); });

// a file dropped onto the shapes column goes straight on: the upload window opens with it picked and continues
const side = $(".side");
const hasFiles = (ev) => [...(ev.dataTransfer?.types || [])].includes("Files");
side.addEventListener("dragover", (ev) => { if (!hasFiles(ev)) return; ev.preventDefault(); side.classList.add("over"); });
side.addEventListener("dragleave", (ev) => { if (!side.contains(ev.relatedTarget)) side.classList.remove("over"); });
side.addEventListener("drop", (ev) => {
  if (!hasFiles(ev)) return;
  ev.preventDefault();
  side.classList.remove("over");
  const file = ev.dataTransfer.files[0];
  openUpload();
  pick(file);
  if (pending?.file === file) $("#upload-next").click();
});

$("#upload-next").addEventListener("click", () => {
  if (!pending) return;
  if (local) return computeLocal();
  if (session) return send();
  if (!sb) {                                                 // not connected yet: no sign-in, just "under development"
    track("upload/dev");
    $("#dev-file").textContent = pending.file.name;
    return show("dev");
  }
  $("#signin-file").textContent = pending.file.name;
  show("signin");
});
$("#signin-back").addEventListener("click", () => { showPicked(); show("pick"); });
$("#dev-back").addEventListener("click", () => { showPicked(); show("pick"); });
$("#done-again").addEventListener("click", openUpload);

// OAuth buttons, one per provider switched on
for (const p of PROVIDERS) {
  const b = document.createElement("button");
  b.type = "button";
  b.className = "provider";
  b.innerHTML = `${ICON[p] || ""}<span>continue with ${LABEL[p] || p}</span>`;
  b.addEventListener("click", async () => {
    if (!sb) return;
    track(`upload/signin/${p}`);
    note("#signin-note", `opening ${LABEL[p] || p}…`);
    if (pending) await stash(pending);
    const { error } = await sb.auth.signInWithOAuth({ provider: p, options: { redirectTo: location.href.split("#")[0] } });
    if (error) note("#signin-note", "That did not work, please try again.");
  });
  $("#providers").append(b);
}

// e-mail: a 6-digit code (stays on the page); the same mail also carries a link
$("#email-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  if (!sb) return;
  const email = ev.target.email.value.trim();
  track("upload/signin/email");
  note("#signin-note", "sending the code…");
  if (pending) await stash(pending);                          // in case the link in the mail is used instead
  const { error } = await sb.auth.signInWithOtp({ email, options: { emailRedirectTo: location.href.split("#")[0] } });
  if (error) return note("#signin-note", error.status === 429 ? "Too many tries, please wait a minute." : "That did not work, please check the address.");
  $("#code-form").hidden = false;
  $("#code-form").code.focus();
  note("#signin-note", `We sent a code to ${email}.`);
});

$("#code-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const email = $("#email-form").email.value.trim(), token = ev.target.code.value.replace(/\D/g, "");
  note("#signin-note", "checking…");
  const { data, error } = await sb.auth.verifyOtp({ email, token, type: "email" });
  if (error) return note("#signin-note", "The code is wrong or expired.");
  session = data.session;
  renderAccount();
  ev.target.reset();
  ev.target.hidden = true;
  if (pending) send();
});

async function send() {
  if (!sb || !session || !pending) return;
  const btn = $("#upload-next"), { file } = pending;
  show("pick");
  btn.disabled = true;
  btn.textContent = "sending…";
  note("#pick-note", "");
  const id = crypto.randomUUID();
  const safe = file.name.replace(/[^\w.-]+/g, "_").slice(-100);
  const path = `${session.user.id}/${id}/${safe}`;
  // the row first: it carries the rate limit, and the bucket only takes files under a row of the same user
  let { error } = await sb.from("uploads").insert({ id, file_name: file.name.slice(0, 200), file_path: path, file_size: file.size });
  if (!error) ({ error } = await sb.storage.from("uploads").upload(path, file, { contentType: file.type || "application/octet-stream" }));
  if (error) {
    btn.disabled = false;
    btn.textContent = "send";
    return note("#pick-note", /limit/i.test(error.message) ? "You sent the most shapes we take per day, please try again tomorrow." : "Sending did not work, please try again.");
  }
  track("upload/sent");
  $("#done-file").textContent = file.name;
  $("#done-email").textContent = session.user.email;
  pending = null;
  dropStash();
  show("done");
  loadUploads();
}

// ------------------------------------------------------------------------------------------------ profile, top left
const account = $("#account"), menu = $("#account-menu"), btn = $("#account-btn");

function renderAccount() {
  account.hidden = !session;
  if (!session) return closeMenu();
  const u = session.user, m = u.user_metadata || {};
  const name = m.full_name || m.name || m.user_name || u.email.split("@")[0];
  $("#avatar").textContent = name.trim()[0].toUpperCase();
  $("#account-name").textContent = name.split(" ")[0];
  $("#account-email").textContent = u.email;
  showPicked();
}

function openMenu() { menu.hidden = false; btn.setAttribute("aria-expanded", "true"); loadUploads(); }
function closeMenu() { menu.hidden = true; btn.setAttribute("aria-expanded", "false"); }
btn.addEventListener("click", () => (menu.hidden ? openMenu() : closeMenu()));
document.addEventListener("click", (ev) => { if (!account.contains(ev.target)) closeMenu(); });
document.addEventListener("keydown", (ev) => { if (ev.key === "Escape" && !menu.hidden) { closeMenu(); btn.focus(); } });
$("#menu-upload").addEventListener("click", () => { closeMenu(); openUpload(); });
$("#sign-out").addEventListener("click", async () => { closeMenu(); await sb?.auth.signOut(); });

async function loadUploads() {
  if (!sb || !session) return;
  const { data, error } = await sb.from("uploads").select("file_name, status, result_url, created_at")
    .order("created_at", { ascending: false }).limit(20);
  const list = $("#uploads-list");
  list.replaceChildren();
  if (error) return list.append(Object.assign(document.createElement("li"), { className: "empty", textContent: "could not load your shapes" }));
  if (!data.length) return list.append(Object.assign(document.createElement("li"), { className: "empty", textContent: "none yet" }));
  for (const r of data) {
    const li = document.createElement("li"), name = document.createElement("span"), st = document.createElement(r.result_url ? "a" : "span");
    name.className = "file";
    name.textContent = r.file_name;
    name.title = `sent ${new Date(r.created_at).toLocaleString()}`;
    st.className = `st ${r.status}`;
    st.textContent = r.result_url ? "open" : STATUS[r.status] || r.status;
    if (r.result_url) st.href = r.result_url;
    li.append(name, st);
    list.append(li);
  }
}

// ------------------------------------------------------------------------------------------------ start
$("#upload").addEventListener("click", openUpload);

async function start() {
  if (!sb) return;
  // back from Google / GitHub / the e-mail link: ?code= (or ?error=) is ours, trade it for a session and tidy the URL
  const url = new URL(location.href), code = url.searchParams.get("code"), failed = url.searchParams.get("error_description");
  if (code || failed) {
    for (const k of ["code", "error", "error_code", "error_description"]) url.searchParams.delete(k);
    history.replaceState(null, "", url.pathname + url.search + url.hash);
  }
  sb.auth.onAuthStateChange((event, s) => { session = s; renderAccount(); });
  if (code) {
    const { error } = await sb.auth.exchangeCodeForSession(code);
    if (error) console.warn("sign-in:", error.message);
  }
  ({ data: { session } } = await sb.auth.getSession());
  renderAccount();
  if (code || failed) {                                        // pick up the file picked before the redirect
    pending = pending || (await unstash()) || null;
    if (pending) {
      openUpload();
      if (session) send();
      else note("#pick-note", "Signing in did not work, please try again.");
    }
  }
}
start();

// CFD Anything - viewer of the precomputed flow_v01 predictions in data/ (written by website/pipeline/predict.py, whose
// pack_frame / pack_vel / write_mesh document the binary layouts). Frames exist for discrete (yaw, pitch, Re): the body
// is tilted nose up by pitch about y, then turned by yaw about z; the flow always comes from the left (+x). On top, the
// body can be rolled about the flow axis x: in free stream that turns the whole solution with it, so the shown frames
// are drawn rolled (body and lines together) and nothing new is needed; the velocity plane stays facing the camera
// (each frame stores planes through x every 15 deg of roll). The drag is a trackball.
// "smooth turning" (on by default; in between it is interpolated, not computed) blends the nearest frames (up to four):
// surface Cp, velocity plane and forces linearly, streamlines by morphing the lines that start from the same seed
// point by point (they fade where the frames disagree, e.g. a line passing above the body in one frame and below it in
// the next); the short wake lines cross-fade. Switched off, only computed frames are shown: the body turns to the
// computed orientation closest to the drag.
// Flow lines switch on and off on their own; the colour field is velocity (|U| on the plane y = 0 facing the camera,
// whatever the roll), pressure or none.
// URL: ?shape=cow&yaw=35&pitch=20&roll=45&re=500000&mode=velocity&lines=0&style=engineering&smooth=0
import * as THREE from "three";
import { toCreasedNormals } from "three/addons/utils/BufferGeometryUtils.js";

const DATA = "data/";
const DEG = Math.PI / 180;
const $ = (sel) => document.querySelector(sel);
// usage events for GoatCounter (vendor/goatcounter/count.js, loaded in index.html); no-op when it is blocked or on
// localhost. once() counts an event at most once per page view (turning, zooming).
const track = (name) => { try { window.goatcounter?.count?.({ path: name, title: name, event: true }); } catch { /* ignore */ } };
const tracked = new Set();
const once = (name) => { if (!tracked.has(name)) { tracked.add(name); track(name); } };
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
const STYLES = { paper: 0, engineering: 1 };
const FIELDS = ["velocity", "pressure"];                    // colour fields; the flow lines toggle independently
const PAPER = "vec3(0.953, 0.949, 0.933)";                  // --bg of the paper style
const U_TOP = 1.6;                                           // top of the |U| / U_inf colour scale (legend in index.html)

// ------------------------------------------------------------------------------------------------ data files
async function fetchOk(url, init) {
  const r = await fetch(url, init);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return r;
}
const getJSON = async (url) => (await fetchOk(url, { cache: "no-cache" })).json();    // builds keep adding frames
const getBuffer = async (url) => (await fetchOk(url)).arrayBuffer();

function parseMesh(buf) {
  const [nv, nf] = new Uint32Array(buf, 0, 2);
  return { position: new Float32Array(buf, 8, 3 * nv), index: new Uint32Array(buf, 8 + 12 * nv, 3 * nf) };
}

function parseFrame(buf) {
  let off = 0;
  const take = (Type, n) => {
    const a = new Type(buf, off, n);
    off += a.byteLength + ((4 - (a.byteLength % 4)) % 4);
    return a;
  };
  const [nLines, nPoints, nVerts, sx, sy] = take(Uint32Array, 6);
  return {
    nLines, nPoints, sx, sy,
    counts: take(Uint16Array, nLines), flags: take(Uint8Array, nLines), q: take(Uint16Array, 3 * nPoints),
    speed: take(Uint8Array, nPoints), cp: take(Uint8Array, nVerts), slice: take(Uint8Array, 2 * sx * sy),
  };
}

function parseVel(buf) {                                     // -> planes x nz x nx speeds (see pack_vel)
  const [nx, nz] = new Uint32Array(buf, 0, 2);
  if (buf.byteLength === 8 + 2 * nx * nz) {                  // before 2026-09-11: the plane y = 0 with a mask channel
    const d = new Uint8Array(buf, 8), data = new Uint8Array(nx * nz);
    for (let i = 0; i < data.length; i++) data[i] = d[2 * i];
    return { nx, nz, planes: 1, data };
  }
  const planes = new Uint32Array(buf, 8, 1)[0];
  return { nx, nz, planes, data: new Uint8Array(buf, 12, planes * nx * nz) };
}

// ------------------------------------------------------------------------------------------------ shaders
// paper: Cp muted blue / paper / muted red around 0; speed warm (slower) / paper (free stream) / teal (faster).
// engineering: turbo (Google's polynomial fit) for both.
const GLSL_COLORS = /* glsl */ `
  vec3 cpPaper(float cp) {
    vec3 paper = ${PAPER};
    if (cp < 0.0) {
      float t = clamp(-cp / 1.5, 0.0, 1.0);
      return t < 0.5 ? mix(paper, vec3(0.55, 0.65, 0.79), 2.0 * t) : mix(vec3(0.55, 0.65, 0.79), vec3(0.25, 0.37, 0.58), 2.0 * t - 1.0);
    }
    float t = clamp(cp, 0.0, 1.0);
    return t < 0.5 ? mix(paper, vec3(0.86, 0.63, 0.56), 2.0 * t) : mix(vec3(0.86, 0.63, 0.56), vec3(0.69, 0.29, 0.21), 2.0 * t - 1.0);
  }
  vec3 speedPaper(float u) {
    vec3 paper = ${PAPER};
    if (u < 1.0) {
      float t = clamp(1.0 - u, 0.0, 1.0);
      return t < 0.5 ? mix(paper, vec3(0.91, 0.70, 0.56), 2.0 * t) : mix(vec3(0.91, 0.70, 0.56), vec3(0.69, 0.33, 0.17), 2.0 * t - 1.0);
    }
    float t = clamp((u - 1.0) / 0.6, 0.0, 1.0);
    return t < 0.5 ? mix(paper, vec3(0.62, 0.81, 0.80), 2.0 * t) : mix(vec3(0.62, 0.81, 0.80), vec3(0.18, 0.55, 0.58), 2.0 * t - 1.0);
  }
  vec3 turbo(float x) {
    x = clamp(x, 0.0, 1.0);
    vec4 v4 = vec4(1.0, x, x * x, x * x * x);
    vec2 v2 = v4.zw * v4.z;
    return vec3(dot(v4, vec4(0.13572138, 4.61539260, -42.66032258, 132.13108234)) + dot(v2, vec2(-152.94239396, 59.28637943)),
                dot(v4, vec4(0.09140261, 2.19418839, 4.84296658, -14.18503333)) + dot(v2, vec2(4.27729857, 2.82956604)),
                dot(v4, vec4(0.10667330, 12.64194608, -60.58204836, 110.36276771)) + dot(v2, vec2(-89.90310912, 27.34824973)));
  }
  vec3 cpColor(float cp, float theme) { return theme < 0.5 ? cpPaper(cp) : turbo((cp + 1.5) / 2.5); }
  vec3 speedColor(float u, float theme) { return theme < 0.5 ? speedPaper(u) : turbo(u / ${U_TOP.toFixed(2)}); }`;

// body: pencil shading on paper (grid in body coordinates, dark rim) or a metallic CAD look; surface Cp in pressure mode.
// uFlat (the shape thumbnails on paper): two flat paper tones, lit and shade, no grid, no rim.
const bodyMaterial = (theme) => new THREE.ShaderMaterial({
  uniforms: { uMode: { value: 0 }, uTheme: { value: theme }, uGrid: { value: 0.06 }, uFlat: { value: 0 }, uPending: { value: 0 },
              uLight: { value: new THREE.Vector3(-0.45, -0.6, 0.66).normalize() } },
  vertexShader: /* glsl */ `
    attribute float aCp;
    varying vec3 vN, vNo, vObj, vView;
    varying float vCp;
    void main() {
      vObj = position; vNo = normal; vCp = aCp;
      vN = normalize(mat3(modelMatrix) * normal);
      vec4 wp = modelMatrix * vec4(position, 1.0);
      vView = cameraPosition - wp.xyz;
      gl_Position = projectionMatrix * viewMatrix * wp;
    }`,
  fragmentShader: /* glsl */ `
    uniform float uMode, uTheme, uGrid, uFlat, uPending;
    uniform vec3 uLight;
    varying vec3 vN, vNo, vObj, vView;
    varying float vCp;
    ${GLSL_COLORS}
    vec3 pending(vec3 col) { return mix(vec3(dot(col, vec3(0.333))), vec3(0.70, 0.695, 0.68), 0.55); }   // upload computing
    void main() {
      vec3 n = normalize(vN) * (gl_FrontFacing ? 1.0 : -1.0), v = normalize(vView);
      float diff = max(dot(n, uLight), 0.0), sky = 0.5 + 0.5 * n.z, rim = 1.0 - abs(dot(n, v));
      vec3 col;
      if (uTheme < 0.5 && uFlat > 0.5) {
        float light = 0.22 + 0.5 * diff + 0.28 * sky, w = max(fwidth(light), 1e-3);
        col = mix(vec3(0.87, 0.86, 0.83), vec3(0.985, 0.98, 0.965), smoothstep(0.62 - w, 0.62 + w, light));
      } else if (uTheme < 0.5) {
        float light = 0.22 + 0.5 * diff + 0.28 * sky;
        vec3 paper = mix(${PAPER}, vec3(1.0), 0.4), graphite = vec3(0.24, 0.235, 0.225);
        col = uMode < 0.5 ? mix(graphite, paper, 0.36 + 0.6 * light) : cpColor(vCp, 0.0) * (0.76 + 0.28 * light);
        vec3 g = vObj / uGrid;
        vec3 d = abs(fract(g - 0.5) - 0.5) / max(fwidth(g), vec3(1e-4));
        vec3 keep = 1.0 - smoothstep(0.8, 0.98, abs(normalize(vNo)));    // no isolines where the surface faces that axis
        vec3 l = (1.0 - min(d, 1.0)) * keep;
        col = mix(col, graphite, 0.14 * max(max(l.x, l.y), l.z));
        col = mix(col, graphite, 0.45 * smoothstep(0.6, 0.98, rim));
      } else {
        vec3 base = uMode < 0.5 ? vec3(0.74, 0.77, 0.81) : cpColor(vCp, 1.0);
        float spec = pow(max(dot(n, normalize(uLight + v)), 0.0), 48.0);
        col = base * (0.28 + 0.6 * diff + 0.22 * sky) + (uMode < 0.5 ? 0.3 : 0.12) * spec;
        col = mix(col, vec3(0.08, 0.11, 0.15), 0.3 * smoothstep(0.7, 1.0, rim));
      }
      if (uPending > 0.5) col = pending(col);
      gl_FragColor = vec4(col, 1.0);
    }`,
});

// inverted hull: back faces pushed out by a constant number of pixels -> outline
const OUTLINE = [new THREE.Color(0.26, 0.255, 0.245), new THREE.Color(0.09, 0.12, 0.16)];
const outlineMaterial = (theme) => new THREE.ShaderMaterial({
  side: THREE.BackSide,
  uniforms: { uWidth: { value: 1.4 }, uRes: { value: new THREE.Vector2(1, 1) }, uColor: { value: OUTLINE[theme].clone() } },
  vertexShader: /* glsl */ `
    uniform float uWidth;
    uniform vec2 uRes;
    void main() {
      vec4 clip = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      vec2 dir = (projectionMatrix * vec4(normalize(normalMatrix * normal), 0.0)).xy;
      clip.xy += normalize(dir + 1e-6) * uWidth * 2.0 / uRes * clip.w;
      gl_Position = clip;
    }`,
  fragmentShader: /* glsl */ `uniform vec3 uColor; void main() { gl_FragColor = vec4(uColor, 1.0); }`,
});

// streamlines: muted blue ink on paper, speed-coloured (same scale as the velocity view) in engineering; travelling
// dashes show the direction and, scaled with the flow speed, how fast it goes; aFade = blend confidence / weight
const lineMaterial = new THREE.ShaderMaterial({
  transparent: true, depthWrite: false,
  uniforms: { uTime: { value: 0 }, uTheme: { value: 0 }, uPlain: { value: 0 }, uDepth: { value: new THREE.Vector2(3, 7) } },
  vertexShader: /* glsl */ `
    attribute float aSpeed, aArc, aLen, aSeed, aFlag, aFade;
    varying float vSpeed, vArc, vLen, vSeed, vFlag, vFade, vDepth;
    void main() {
      vSpeed = aSpeed; vArc = aArc; vLen = aLen; vSeed = aSeed; vFlag = aFlag; vFade = aFade;
      vec4 mv = modelViewMatrix * vec4(position, 1.0);
      vDepth = -mv.z;
      gl_Position = projectionMatrix * mv;
    }`,
  fragmentShader: /* glsl */ `
    uniform float uTime, uTheme, uPlain;
    uniform vec2 uDepth;
    varying float vSpeed, vArc, vLen, vSeed, vFlag, vFade, vDepth;
    ${GLSL_COLORS}
    void main() {
      float ph = fract(vArc * 1.4 - uTime + vSeed);
      float pulse = smoothstep(0.0, 0.06, ph) * (1.0 - smoothstep(0.06, 0.3, ph));
      float fade = vFade * smoothstep(0.0, 0.25, vArc) * smoothstep(0.0, 0.25, vLen - vArc)
                 * mix(1.0, 0.4, clamp((vDepth - uDepth.x) / (uDepth.y - uDepth.x), 0.0, 1.0));
      vec3 col;
      float a;
      if (uTheme < 0.5) {
        col = mix(vec3(0.60, 0.69, 0.80), vec3(0.25, 0.41, 0.64), clamp((vSpeed - 0.55) / 0.75, 0.0, 1.0));
        col = mix(col, vec3(0.20, 0.32, 0.53), 0.65 * vFlag);
        a = 0.45 + 0.5 * pulse;
      } else {
        col = speedColor(vSpeed, 1.0) * 0.85;               // a little darker, for the white background
        a = 0.55 + 0.4 * pulse;
      }
      if (uPlain > 0.5) {                                    // over a colour field: neutral ink, so the field reads
        col = uTheme < 0.5 ? vec3(0.22, 0.24, 0.28) : vec3(0.1, 0.12, 0.15);
        a *= 0.55;
      }
      gl_FragColor = vec4(col, a * fade);
    }`,
});

// velocity plane y = 0, facing the camera: texture R = |U| / U_inf over 0 .. speed_max (see rollVel); no mask, no
// iso-lines, so the flow lines read
const velMaterial = new THREE.ShaderMaterial({
  transparent: true, depthWrite: false, side: THREE.DoubleSide,
  uniforms: { uTex: { value: null }, uMax: { value: 2 }, uTheme: { value: 0 }, uFacing: { value: 1 } },
  vertexShader: /* glsl */ `varying vec2 vUv; void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
  fragmentShader: /* glsl */ `
    uniform sampler2D uTex;
    uniform float uMax, uTheme, uFacing;
    varying vec2 vUv;
    ${GLSL_COLORS}
    void main() {
      vec4 t = texture2D(uTex, vUv);
      float u = t.r * uMax;
      vec3 col = speedColor(u, uTheme);
      float edge = smoothstep(0.0, 0.06, vUv.x) * smoothstep(0.0, 0.2, 1.0 - vUv.x) * smoothstep(0.0, 0.08, vUv.y) * smoothstep(0.0, 0.08, 1.0 - vUv.y);
      // the free stream (u = 1) is fully transparent. paper: the first half of each ramp mixes paper into the colour,
      // so that mix becomes opacity instead (the same look over the page, see-through where lines or the grid are behind)
      float strength;
      if (uTheme < 0.5) {
        strength = clamp(2.0 * (u < 1.0 ? 1.0 - u : (u - 1.0) / 0.6), 0.0, 1.0);
        col = (col - ${PAPER} * (1.0 - strength)) / max(strength, 1e-3);
      } else {
        strength = smoothstep(0.03, 0.3, abs(u - 1.0));
      }
      float facing = smoothstep(0.12, 0.4, uFacing);          // rolled edge-on it would only be a smear
      // no fluid mask: the opaque body hides the plane inside it, and body texels carry the nearest fluid value, so
      // the colour runs right up to the wall instead of fading out a texel early (a paper halo round the body)
      gl_FragColor = vec4(col, 0.92 * strength * edge * facing);
    }`,
});

// ------------------------------------------------------------------------------------------------ body
function makeBody(entry, theme) {
  const outline = new THREE.BufferGeometry();
  outline.setAttribute("position", new THREE.BufferAttribute(entry.mesh.position, 3));
  outline.setIndex(new THREE.BufferAttribute(entry.mesh.index, 1));
  outline.computeVertexNormals();                            // smooth normals for the outline shell
  outline.computeBoundingSphere();
  const shaded = toCreasedNormals(outline, 50 * DEG);        // non-indexed, corner k belongs to vertex index[k]
  shaded.setAttribute("aCp", new THREE.BufferAttribute(new Float32Array(entry.mesh.index.length), 1));
  const body = new THREE.Mesh(shaded, bodyMaterial(theme));
  const hull = new THREE.Mesh(outline, outlineMaterial(theme));
  const group = new THREE.Group();
  group.add(hull, body);
  return { group, body, hull, entry, radius: outline.boundingSphere.radius };
}

function disposeBody(obj) {
  obj.body.geometry.dispose();
  obj.hull.geometry.dispose();
  obj.body.material.dispose();
  obj.hull.material.dispose();
}

// ------------------------------------------------------------------------------------------------ orientations
const angleDiff = (a, b) => ((((a - b) % 360) + 540) % 360) - 180;     // signed, in [-180, 180)
const norm360 = (a) => ((a % 360) + 360) % 360;
const smoothstep = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };

function hash01(i) {
  const x = Math.sin(i * 12.9898 + 4.1) * 43758.5453;
  return x - Math.floor(x);
}

function rotation(yaw, pitch) {                              // body -> world: Rz(yaw) Ry(pitch), as in predict.py
  const a = yaw * DEG, b = pitch * DEG, ca = Math.cos(a), sa = Math.sin(a), cb = Math.cos(b), sb = Math.sin(b);
  return [[ca * cb, -sa, ca * sb], [sa * cb, ca, sa * sb], [-sb, 0, cb]];
}

const transposeTimes = (A, B) =>                             // A^T B
  [0, 1, 2].map((i) => [0, 1, 2].map((j) => A[0][i] * B[0][j] + A[1][i] * B[1][j] + A[2][i] * B[2][j]));

function fileOrientation(file) {                             // f030_p+30_re200000.bin -> {yaw: 30, pitch: 30}
  const m = /^f(\d{3})(?:_p([+-]\d+))?(?:_re\d+)?\.bin$/.exec(file);
  return m ? { yaw: +m[1], pitch: m[2] ? +m[2] : 0 } : null;
}

function nearestVertexFinder(p) {                            // spatial hash over the render-mesh vertices
  const h = 0.04, cells = new Map();
  const cell = (x) => Math.floor(x / h);
  const key = (i, j, k) => (i * 73856093) ^ (j * 19349663) ^ (k * 83492791);
  for (let v = 0; v < p.length / 3; v++) {
    const kk = key(cell(p[3 * v]), cell(p[3 * v + 1]), cell(p[3 * v + 2]));
    if (!cells.has(kk)) cells.set(kk, []);
    cells.get(kk).push(v);
  }
  return (x, y, z) => {
    const ci = cell(x), cj = cell(y), ck = cell(z);
    for (let r = 1; r <= 8; r++) {
      let best = -1, bd = Infinity;
      for (let i = ci - r; i <= ci + r; i++) for (let j = cj - r; j <= cj + r; j++) for (let k = ck - r; k <= ck + r; k++) {
        for (const v of cells.get(key(i, j, k)) || []) {
          const d = (p[3 * v] - x) ** 2 + (p[3 * v + 1] - y) ** 2 + (p[3 * v + 2] - z) ** 2;
          if (d < bd) { bd = d; best = v; }
        }
      }
      if (best >= 0) return best;
    }
    return 0;
  };
}

// A frame computed for another, symmetric orientation (sphere, cube) holds the surface Cp of that orientation's
// vertices: each vertex takes the value of the vertex that sits at its place there.
function cpMap(entry, f) {
  const src = fileOrientation(f.file);
  if (!src || (Math.abs(angleDiff(src.yaw, f.yaw)) < 1e-6 && src.pitch === f.pitch)) return null;
  const key = `${f.file}|${f.yaw}|${f.pitch}`;
  entry.maps ??= new Map();
  if (!entry.maps.has(key)) {
    const M = transposeTimes(rotation(src.yaw, src.pitch), rotation(f.yaw, f.pitch)), p = entry.mesh.position;
    entry.nearest ??= nearestVertexFinder(p);
    const map = new Uint32Array(p.length / 3);
    for (let i = 0; i < map.length; i++) {
      const x = p[3 * i], y = p[3 * i + 1], z = p[3 * i + 2];
      map[i] = entry.nearest(M[0][0] * x + M[0][1] * y + M[0][2] * z, M[1][0] * x + M[1][1] * y + M[1][2] * z,
                             M[2][0] * x + M[2][1] * y + M[2][2] * z);
    }
    entry.maps.set(key, map);
  }
  return entry.maps.get(key);
}

function prepareFrames(meta) {                               // frames of older builds carry no pitch / Re
  for (const f of meta.frames) {
    f.pitch ??= 0;
    f.re ??= meta.re;
  }
  meta.res = [...new Set(meta.frames.map((f) => f.re))].sort((a, b) => a - b);
  meta.rows = new Map();                                     // "re|pitch" -> [{f, i}] sorted by yaw
  meta.pitches = new Map();                                  // re -> sorted pitches
  meta.frames.forEach((f, i) => {
    const k = `${f.re}|${f.pitch}`;
    if (!meta.rows.has(k)) meta.rows.set(k, []);
    meta.rows.get(k).push({ f, i });
  });
  for (const row of meta.rows.values()) row.sort((a, b) => a.f.yaw - b.f.yaw);
  for (const re of meta.res) {
    meta.pitches.set(re, [...new Set(meta.frames.filter((f) => f.re === re).map((f) => f.pitch))].sort((a, b) => a - b));
  }
}

const nearestRe = (meta, re) => meta.res.reduce((b, x) => (Math.abs(Math.log(x / re)) < Math.abs(Math.log(b / re)) ? x : b), meta.res[0] ?? meta.re);

function pitchRange(meta, re) {                              // an upload without frames yet: level only
  const ps = meta.pitches.get(nearestRe(meta, re)) ?? [0];
  return [ps[0], ps[ps.length - 1]];
}

// Weights of the computed frames around (yaw, pitch) at the closest Re: linear in pitch between the two tilt rows
// around it, and in each row linear in yaw between the two angles around it (rows can have different yaw steps).
function blendWeights(meta, re, yaw, pitch) {
  const r = nearestRe(meta, re), ps = meta.pitches.get(r);
  const p = Math.min(ps[ps.length - 1], Math.max(ps[0], pitch));
  let j = 0;
  while (j < ps.length - 1 && ps[j + 1] <= p) j++;
  const p0 = ps[j], p1 = ps[Math.min(j + 1, ps.length - 1)];
  const tp = p1 > p0 ? (p - p0) / (p1 - p0) : 0;
  const y = norm360(yaw), acc = new Map();
  for (const [pr, wp] of [[p0, 1 - tp], [p1, tp]]) {
    if (wp <= 1e-4) continue;
    const row = meta.rows.get(`${r}|${pr}`);
    let a = row.length - 1;                                  // last angle at or below y (wraps to the end)
    for (let q = 0; q < row.length; q++) if (row[q].f.yaw <= y + 1e-9) a = q;
    const b = (a + 1) % row.length;
    const span = norm360(row[b].f.yaw - row[a].f.yaw) || 360;
    const ty = row.length > 1 ? norm360(y - row[a].f.yaw) / span : 0;
    for (const [q, w] of [[a, wp * (1 - ty)], [b, wp * ty]]) {
      if (w > 1e-4) acc.set(row[q].i, (acc.get(row[q].i) || 0) + w);
    }
  }
  return [...acc].map(([k, w]) => ({ k, w }));
}

function nearestFrame(meta, re, yaw, pitch) {                // for the prefetch: the closest computed frame
  return blendWeights(meta, re, yaw, pitch).reduce((a, b) => (b.w > a.w ? b : a)).k;
}

// Full orientation: q = Rx(roll) Rz(yaw) Ry(pitch), pitch in [-90, 90] like the computed frames. (yaw, pitch) only
// depend on the inflow direction in body coordinates; the roll about x is drawn.
const X_AXIS = new THREE.Vector3(1, 0, 0), Y_AXIS = new THREE.Vector3(0, 1, 0), Z_AXIS = new THREE.Vector3(0, 0, 1);
const axisQuat = (axis, deg) => new THREE.Quaternion().setFromAxisAngle(axis, deg * DEG);
const orientQuat = (yaw, pitch, roll = 0) => axisQuat(X_AXIS, roll).multiply(axisQuat(Z_AXIS, yaw)).multiply(axisQuat(Y_AXIS, pitch));

function quatMat({ x, y, z, w }) {                           // row-major rotation matrix
  return [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
          [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
          [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]];
}

function bestRoll(M, R) {                                    // roll about x that brings R closest to M (degrees)
  const n = (i, j) => M[i][0] * R[j][0] + M[i][1] * R[j][1] + M[i][2] * R[j][2];     // (M R^T)_ij
  return Math.atan2(n(2, 1) - n(1, 2), n(1, 1) + n(2, 2)) / DEG;
}

// q -> {yaw, pitch, roll}. Ties go to the previous orientation: at the poles (flow along the body's y axis, where pitch
// and roll turn about the same axis) it keeps its pitch, on the seam pitch = ±90 its branch.
function decompose(q, prev) {
  const M = quatMat(q), [dx, dy, dz] = M[0];                 // inflow direction in body coordinates
  const h = Math.hypot(dx, dz);
  let o;
  if (h < 1e-7) {
    o = { yaw: dy < 0 ? 90 : -90, pitch: prev?.pitch ?? 0 };
  } else {
    const cands = [1, -1].filter((s) => s * dx >= -1e-9)
      .map((s) => ({ yaw: Math.atan2(-dy, s * h) / DEG, pitch: Math.atan2(s * dz, s * dx) / DEG }));
    const dist = (c) => (prev ? Math.abs(angleDiff(c.yaw, prev.yaw)) + Math.abs(c.pitch - prev.pitch) : 0);
    o = cands.reduce((a, b) => (dist(b) < dist(a) ? b : a));
  }
  o.roll = bestRoll(M, rotation(o.yaw, o.pitch));
  return o;
}

// (yaw, pitch) of a computed frame, rolled to come closest to q
const frameQuat = (yaw, pitch, q) => orientQuat(yaw, pitch, bestRoll(quatMat(q), rotation(yaw, pitch)));

// ------------------------------------------------------------------------------------------------ viewer
const stage = $("#stage"), canvas = $("#view"), statusEl = $("#status"), speedEl = $("#speed");
// qTo (oriTo = its yaw / pitch / roll): where the drag and the arrow keys put the body; q (ori): the body on screen.
// Smooth turning: the body follows the drag exactly (arrow keys glide) and the flow is blended around it. Otherwise the
// flow is the computed frame closest to oriTo and the body turns to the frame on screen (shownFrame, rolled as close to
// qTo as it gets), so body and flow always belong together and nothing turns back. flowRoll: the roll the lines and
// the velocity plane are drawn with.
const state = { shapes: [], current: null, lines: true, field: null, style: "paper", re: 5e5, smooth: true,
                q: new THREE.Quaternion(), qTo: new THREE.Quaternion(), ori: { yaw: 0, pitch: 0, roll: 0 },
                oriTo: { yaw: 0, pitch: 0, roll: 0 }, flowRoll: 0, force: null, reShown: 5e5, drawn: false,
                prefetchedAround: -1, shownFrame: null, lastSig: null };
const cache = new Map(), ready = new Map(), failed = new Map();   // key -> promise / parsed file / time of failure

let renderer;
try {
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
} catch (err) {
  setStatus("this demo needs WebGL, which is not available in this browser", true);
  throw err;
}
renderer.setClearColor(0x000000, 0);
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));   // sharp enough for thin lines, a third of 2x fill

const scene = new THREE.Scene();
const FOV = 28;                                              // vertical field of view at zoom 1
const camera = new THREE.PerspectiveCamera(FOV, 1, 0.05, 60);
camera.up.set(0, 0, 1);
const TARGET = new THREE.Vector3(0.3, 0, -0.05);            // at zoom 1 the view centre sits a little behind the body
const target = new THREE.Vector3();
const AZIMUTH = -90 * DEG, TILT = 16 * DEG;                 // fixed camera across the flow: it always runs left to right
// Zoom narrows the field of view (the camera stays put, so nothing near it blows up) and, from 1x to 2x, moves the
// centre onto the body. No panning; zoom 1 is the widest view and shows all computed data.
const ZOOM_MIN = 1, ZOOM_MAX = 3;
let baseDist = 5, zoom = 1, zoomTo = 1;
const clampZoom = (z) => Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, z));

const lines = new THREE.LineSegments(new THREE.BufferGeometry(), lineMaterial);
lines.renderOrder = 2;
lines.frustumCulled = false;
let velTex = null, velBuf = null, velAcc = null;             // blended velocity plane, sized like the planes
let velParts = null, velRolled = false, velSlice = null;     // the frames' plane stacks with weights; see rollVel
const velPlane = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), velMaterial);
velPlane.rotation.x = Math.PI / 2;                           // x-z plane, facing the camera
velPlane.renderOrder = 1;
velPlane.visible = false;
const flowGroup = new THREE.Group();                         // the shown frames, turned by the roll about x
flowGroup.add(lines);
scene.add(flowGroup, velPlane);                              // the plane stays facing the camera (see rollVel)
let bodyObj = null;
let dashTime = 0;
let dirty = true;                                            // something on screen changed: draw the next frame
const invalidate = () => { dirty = true; };
let blendDirty = false, blendToken = 0;                      // the blend has to be recomputed
const requestBlend = () => { blendDirty = true; };

function setStatus(text, error = false) {
  statusEl.textContent = text;
  statusEl.classList.toggle("error", error);
  statusEl.classList.toggle("done", !text);
}

function placeCamera() {
  const d = baseDist, t = 1 - smoothstep(1, 2, zoom);
  target.set(TARGET.x * t, TARGET.y * t, TARGET.z * t);
  camera.fov = (2 * Math.atan(Math.tan((FOV * DEG) / 2) / zoom)) / DEG;
  camera.updateProjectionMatrix();
  camera.position.set(target.x + d * Math.cos(TILT) * Math.cos(AZIMUTH), target.y + d * Math.cos(TILT) * Math.sin(AZIMUTH),
                      target.z + d * Math.sin(TILT));
  camera.lookAt(target);
  camera.updateMatrixWorld();
  lineMaterial.uniforms.uDepth.value.set(d - 1.2, d + 1.6);
  invalidate();
}

function resize() {
  const w = stage.clientWidth, h = stage.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  const tv = Math.tan((FOV * DEG) / 2);
  baseDist = Math.max(2.0 / (2 * tv), 3.5 / (2 * tv * camera.aspect));   // at zoom 1 ~3.5 L wide, ~2 L high
  if (bodyObj) bodyObj.hull.material.uniforms.uRes.value.set(w * renderer.getPixelRatio(), h * renderer.getPixelRatio());
  placeCamera();
  drawTriad();                                               // the camera direction never changes, only its distance
}

function zoomBy(f) {
  zoomTo = clampZoom(zoomTo * f);
  saveUrl();
}

function drawTriad() {
  const inv = camera.quaternion.clone().invert();
  const axes = [["x", new THREE.Vector3(1, 0, 0)], ["y", new THREE.Vector3(0, 1, 0)], ["z", new THREE.Vector3(0, 0, 1)]]
    .map(([name, v]) => [name, v.applyQuaternion(inv)])
    .sort((a, b) => a[1].z - b[1].z);                         // far axes first
  $("#triad").innerHTML = axes.map(([name, v]) => {
    const x = 28 * v.x, y = -28 * v.y, op = v.z > 0.5 ? 0.45 : 1;
    const a = Math.atan2(y, x), hx = (s) => x - 5 * Math.cos(a + s), hy = (s) => y - 5 * Math.sin(a + s);
    return `<g opacity="${op}"><line x1="0" y1="0" x2="${x}" y2="${y}"/>` +
      `<line x1="${x}" y1="${y}" x2="${hx(0.45)}" y2="${hy(0.45)}"/><line x1="${x}" y1="${y}" x2="${hx(-0.45)}" y2="${hy(-0.45)}"/>` +
      `<text x="${1.4 * x}" y="${1.4 * y}">${name}</text></g>`;
  }).join("");
}

// ------------------------------------------------------------------------------------------------ loading
const CACHE_MAX = 240;                                       // parsed files kept in memory (~150 kB each)

function load(entry, file, parse) {                          // symmetric bodies share files between orientations
  const key = `${entry.name}/${file}`;
  if (cache.has(key)) {
    const p = cache.get(key);
    cache.delete(key);                                       // most recently used goes to the end
    cache.set(key, p);
    return p;
  }
  if (performance.now() - (failed.get(key) ?? -Infinity) < 30000) return Promise.reject(new Error("recently missing"));
  const p = getBuffer(baseOf(entry) + file).then(parse);
  p.then((v) => ready.set(key, v), () => { cache.delete(key); failed.set(key, performance.now()); });
  cache.set(key, p);
  while (cache.size > CACHE_MAX) {
    const old = cache.keys().next().value;
    cache.delete(old);
    ready.delete(old);
  }
  return p;
}
const baseOf = (entry) => entry.base ?? `${DATA}${entry.name}/`;      // uploads: uploads/<id>/ (local preview)
const peek = (entry, file) => ready.get(`${entry.name}/${file}`);
const hasFailed = (entry, file) => failed.has(`${entry.name}/${file}`);
const velName = (file) => file.replace(/\.bin$/, ".vel.bin");

// Around the closest computed frame: the frames one step away (turn, tilt, flow speed) first, then the rest of its
// tilt row, one file at a time; a newer centre restarts it.
let prefetchRun = 0;
async function prefetch(entry, k0) {
  const run = ++prefetchRun, m = entry.meta, f0 = m.frames[k0];
  const row = m.rows.get(`${f0.re}|${f0.pitch}`).map(({ f }) => f), pos = row.indexOf(f0);
  const near = [1, -1, 2, -2].map((d) => row[(pos + d + row.length) % row.length]);
  const ps = m.pitches.get(f0.re);
  for (const j of [ps.indexOf(f0.pitch) + 1, ps.indexOf(f0.pitch) - 1]) {
    if (j >= 0 && j < ps.length) near.push(m.frames[nearestFrame(m, f0.re, f0.yaw, ps[j])]);
  }
  for (const j of [m.res.indexOf(f0.re) + 1, m.res.indexOf(f0.re) - 1]) {
    if (j >= 0 && j < m.res.length) near.push(m.frames[nearestFrame(m, m.res[j], f0.yaw, f0.pitch)]);
  }
  const rest = [...row].sort((a, b) => Math.abs(angleDiff(a.yaw, f0.yaw)) - Math.abs(angleDiff(b.yaw, f0.yaw)));
  for (const file of [...new Set([...near, ...rest].map((f) => f.file))].slice(0, 24)) {
    if (run !== prefetchRun || state.current !== entry) return;
    try {
      await load(entry, file, parseFrame);
      if (state.field === "velocity") await load(entry, velName(file), parseVel).catch(() => null);
    } catch { /* shown again on demand */ }
  }
}

// ------------------------------------------------------------------------------------------------ blending
// the lines of a frame: inflow lines keyed by their seed (first point; the seeds are the same in every frame of a
// shape), the wake lines listed apart
function frameLines(fr) {
  if (!fr.lines) {
    const inflow = new Map(), wake = [];
    let start = 0;
    for (let l = 0; l < fr.nLines; l++) {
      const c = fr.counts[l];
      if (fr.flags[l]) wake.push([start, c]);
      else inflow.set(fr.q[3 * start + 1] * 65536 + fr.q[3 * start + 2], [start, c]);
      start += c;
    }
    fr.lines = { inflow, wake };
  }
  return fr.lines;
}

const LINE_ATTRS = [["position", 3], ["aSpeed", 1], ["aArc", 1], ["aLen", 1], ["aSeed", 1], ["aFlag", 1], ["aFade", 1]];
let lineCap = 0, lineArr = null, lineIdx = null;

function ensureLineCap(nPoints) {                            // reusable buffers; grown when a blend needs more
  if (nPoints <= lineCap) return;
  lineCap = Math.ceil(nPoints * 1.3);
  const g = new THREE.BufferGeometry();
  lineArr = {};
  for (const [name, size] of LINE_ATTRS) {
    lineArr[name] = new Float32Array(lineCap * size);
    g.setAttribute(name, new THREE.BufferAttribute(lineArr[name], size).setUsage(THREE.DynamicDrawUsage));
  }
  lineIdx = new Uint32Array(lineCap * 2);
  g.setIndex(new THREE.BufferAttribute(lineIdx, 1).setUsage(THREE.DynamicDrawUsage));
  lines.geometry.dispose();
  lines.geometry = g;
}

// Runs on every drag move: plain index loops and preallocated arrays, no per-point allocations.
const QS = [], SPS = [], STS = [], WS = [];

function blendLines(parts, meta) {
  const { lo, hi } = meta.box, sm = meta.speed_max / 255;
  const sx = (hi[0] - lo[0]) / 65535, sy = (hi[1] - lo[1]) / 65535, sz = (hi[2] - lo[2]) / 65535;
  ensureLineCap(parts.reduce((s, p) => s + p.fr.nPoints, 0));
  const P = lineArr.position, S = lineArr.aSpeed, A = lineArr.aArc, Ln = lineArr.aLen;
  const Sd = lineArr.aSeed, F = lineArr.aFlag, Fd = lineArr.aFade;
  const ref = parts.reduce((a, b) => (b.w > a.w ? b : a));
  let n = 0, e = 0;
  // inflow lines of the same seed, morphed point by point; they fade where the frames disagree by more than ~0.1 L
  for (const key of frameLines(ref.fr).inflow.keys()) {
    let m = 0, wsum = 0, c = Infinity;
    for (const p of parts) {
      const L = frameLines(p.fr).inflow.get(key);
      if (!L) continue;
      QS[m] = p.fr.q; SPS[m] = p.fr.speed; STS[m] = L[0]; WS[m] = p.w;
      m++;
      wsum += p.w;
      c = Math.min(c, L[1]);
    }
    for (let t = 0; t < m; t++) WS[t] /= wsum;
    const start = n, rnd = hash01(key);
    let s = 0, px = 0, py = 0, pz = 0;
    for (let j = 0; j < c; j++) {
      let qx = 0, qy = 0, qz = 0, sp = 0;
      for (let t = 0; t < m; t++) {
        const i = STS[t] + j, q = QS[t], w = WS[t];
        qx += w * q[3 * i]; qy += w * q[3 * i + 1]; qz += w * q[3 * i + 2]; sp += w * SPS[t][i];
      }
      let d2 = 0;
      for (let t = 0; m > 1 && t < m; t++) {
        const i = STS[t] + j, q = QS[t];
        const dx = (q[3 * i] - qx) * sx, dy = (q[3 * i + 1] - qy) * sy, dz = (q[3 * i + 2] - qz) * sz;
        d2 = Math.max(d2, dx * dx + dy * dy + dz * dz);
      }
      const x = lo[0] + qx * sx, y = lo[1] + qy * sy, z = lo[2] + qz * sz;
      if (j > 0) {
        s += Math.sqrt((x - px) * (x - px) + (y - py) * (y - py) + (z - pz) * (z - pz));
        lineIdx[e++] = n - 1;
        lineIdx[e++] = n;
      }
      P[3 * n] = px = x; P[3 * n + 1] = py = y; P[3 * n + 2] = pz = z;
      S[n] = sp * sm; A[n] = s; Sd[n] = rnd; F[n] = 0;
      Fd[n] = d2 < 0.0025 ? 1 : 1 - smoothstep(0.05, 0.18, Math.sqrt(d2));
      n++;
    }
    Ln.fill(s, start, n);
  }
  // wake lines: every frame's own, cross-faded with its weight
  for (const p of parts) {
    const q = p.fr.q, speed = p.fr.speed;
    for (const [st, c] of frameLines(p.fr).wake) {
      const start = n, rnd = hash01(st + 7);
      let s = 0, px = 0, py = 0, pz = 0;
      for (let j = 0; j < c; j++) {
        const i = st + j;
        const x = lo[0] + q[3 * i] * sx, y = lo[1] + q[3 * i + 1] * sy, z = lo[2] + q[3 * i + 2] * sz;
        if (j > 0) {
          s += Math.sqrt((x - px) * (x - px) + (y - py) * (y - py) + (z - pz) * (z - pz));
          lineIdx[e++] = n - 1;
          lineIdx[e++] = n;
        }
        P[3 * n] = px = x; P[3 * n + 1] = py = y; P[3 * n + 2] = pz = z;
        S[n] = speed[i] * sm; A[n] = s; Sd[n] = rnd; F[n] = 1; Fd[n] = p.w;
        n++;
      }
      Ln.fill(s, start, n);
    }
  }
  const g = lines.geometry;
  for (const [name, size] of LINE_ATTRS) {
    const at = g.getAttribute(name);
    at.clearUpdateRanges();
    at.addUpdateRange(0, n * size);
    at.needsUpdate = true;
  }
  g.index.clearUpdateRanges();
  g.index.addUpdateRange(0, e);
  g.index.needsUpdate = true;
  g.setDrawRange(0, e);
}

function blendSurface(entry, parts) {
  const nv = entry.mesh.position.length / 3;
  if (entry.cpBlend?.length !== nv) entry.cpBlend = new Float32Array(nv);
  const cpv = entry.cpBlend.fill(0);
  for (const { f, fr, w } of parts) {
    const map = cpMap(entry, f);
    for (let v = 0; v < nv; v++) cpv[v] += w * fr.cp[map ? map[v] : v];
  }
  const a = bodyObj.body.geometry.getAttribute("aCp"), idx = entry.mesh.index, [lo, hi] = entry.meta.cp_range;
  for (let k = 0; k < idx.length; k++) a.array[k] = lo + ((hi - lo) * cpv[idx[k]]) / 254;
  a.needsUpdate = true;
}

function blendVel(entry, parts, token) {
  const vs = [];
  let wsum = 0, missing = 0;
  for (const p of parts) {
    const name = velName(p.f.file), v = peek(entry, name);
    if (v) {
      vs.push([v, p.w]);
      wsum += p.w;
    } else if (hasFailed(entry, name)) {
      missing++;
    } else {
      load(entry, name, parseVel).then(() => { if (token === blendToken) requestBlend(); }, () => { if (token === blendToken) requestBlend(); });
    }
  }
  if (!vs.length) {
    velPlane.visible = false;
    if (missing === parts.length) setStatus("the velocity field of this orientation is still being computed");
    return missing === parts.length;
  }
  const [v0] = vs[0], n = v0.nx * v0.nz;
  if (velBuf?.length !== n) {
    velBuf = new Uint8Array(n);
    velAcc = new Float32Array(n);
  }
  velParts = vs.filter(([v]) => v.nx === v0.nx && v.nz === v0.nz);
  if (!velTex || velTex.image.width !== v0.nx || velTex.image.height !== v0.nz) {
    velTex?.dispose();                                       // WebGL2 texture storage cannot change size
    velTex = new THREE.DataTexture(velBuf, v0.nx, v0.nz, THREE.RedFormat, THREE.UnsignedByteType);
    velTex.unpackAlignment = 1;
    velTex.magFilter = velTex.minFilter = THREE.LinearFilter;
    velMaterial.uniforms.uTex.value = velTex;
  }
  const { lo, hi } = entry.meta.box;                         // planes of a running velslice job come before vel_slice
  velSlice = entry.meta.vel_slice || { y: 0, x: [lo[0], hi[0]], z: [lo[2], hi[2]] };
  velPlane.scale.set(velSlice.x[1] - velSlice.x[0], velSlice.z[1] - velSlice.z[0], 1);
  rollVel();
  velPlane.visible = true;
  return vs.length + missing === parts.length;               // false: planes still loading
}

// The plane facing the camera at the current roll. Plane k of a frame's n holds its flow at theta = k 180 / n through
// the x axis, (x, s sin theta, s cos theta); drawn rolled by phi about x, plane theta = phi is the world plane y = 0
// with s = world z (past 180 deg: plane phi - 180 with s flipped). Between two planes it blends them, texel by texel.
// Files with only the plane y = 0 (built before 2026-09-11) roll with the flow instead, fading out edge-on.
function rollVel() {
  const [v0] = velParts[0], nx = v0.nx, nz = v0.nz;
  velRolled = velParts.every(([v]) => v.planes > 1);
  const phi = velRolled ? norm360(state.flowRoll) : 0;
  velAcc.fill(0);
  let ws = 0;
  for (const [v, w] of velParts) {
    const n = velRolled ? v.planes : 1, t = (phi / 180) * n, j0 = Math.floor(t), a = t - j0;
    ws += w;
    for (const [j, wj] of [[j0 % (2 * n), 1 - a], [(j0 + 1) % (2 * n), a]]) {
      if (wj < 1e-4) continue;
      const src = v.data, base = (j % n) * nx * nz, flip = j >= n, ww = w * wj;
      for (let r = 0; r < nz; r++) {
        const o = base + (flip ? nz - 1 - r : r) * nx, d = r * nx;
        for (let c = 0; c < nx; c++) velAcc[d + c] += ww * src[o + c];
      }
    }
  }
  for (let i = 0; i < velBuf.length; i++) velBuf[i] = Math.round(velAcc[i] / ws);
  velTex.image.data = velBuf;
  velTex.needsUpdate = true;
  const roll = velRolled ? 0 : state.flowRoll * DEG, s = velSlice;
  velPlane.rotation.x = Math.PI / 2 + roll;                  // x-z plane, facing the camera (rolled: legacy files)
  velPlane.position.set((s.x[0] + s.x[1]) / 2, s.y, (s.z[0] + s.z[1]) / 2).applyAxisAngle(X_AXIS, roll);
  const view = camera.getWorldDirection(new THREE.Vector3());        // plane normal: y turned about x
  velMaterial.uniforms.uFacing.value = velRolled ? 1 : Math.abs(Math.cos(roll) * view.y + Math.sin(roll) * view.z);
}

// Recompute what is shown from the frames around the current orientation. Frames not in memory yet are requested;
// each arrival blends again, so the view sharpens as they come in.
function updateBlend() {
  blendDirty = false;
  const entry = state.current, m = entry?.meta;
  if (!m || bodyObj?.entry !== entry) return;
  if (!m.frames.length) return showPending(entry);
  bodyObj.body.material.uniforms.uPending.value = 0;
  const weights = state.smooth ? blendWeights(m, state.re, state.ori.yaw, state.ori.pitch)
                               : [{ k: nearestFrame(m, state.re, state.oriTo.yaw, state.oriTo.pitch), w: 1 }];
  const sig = `${state.field}|${weights.map(({ k, w }) => `${k}:${w.toFixed(4)}`).join(",")}`;
  if (sig === state.lastSig) return;                         // e.g. a drag that stays closest to the same frame
  const token = ++blendToken, parts = [];
  for (const { k, w } of weights) {
    const f = m.frames[k], fr = peek(entry, f.file);
    if (fr) parts.push({ k, f, fr, w });
    else load(entry, f.file, parseFrame).then(() => { if (token === blendToken) requestBlend(); }, () => {});
  }
  if (!parts.length) {
    if (!state.drawn) setStatus("loading the flow…");
    return;
  }
  const wsum = parts.reduce((s, p) => s + p.w, 0);
  for (const p of parts) p.w /= wsum;
  blendLines(parts, m);
  blendSurface(entry, parts);
  updateReadout(parts);
  setStatus("");
  if (entry.upload) showUploadState(entry);
  let complete = parts.length === weights.length;
  if (state.field === "velocity") complete = blendVel(entry, parts, token) && complete;
  state.lastSig = complete ? sig : null;                     // incomplete: blend again as the files arrive
  state.drawn = true;
  invalidate();
  const main = parts.reduce((a, b) => (b.w > a.w ? b : a));
  state.shownFrame = main.f;
  if (main.k !== state.prefetchedAround) {
    state.prefetchedAround = main.k;
    prefetch(entry, main.k);
  }
}

// ------------------------------------------------------------------------------------------------ numbers
const fmt = (x) => `${x < 0 ? "−" : ""}${Math.abs(x).toFixed(2)}`;

// ± next to drag and lift: mean error of flow_v01 on the 87 frozen val cases. Cd: relative
// (8.6 %). Cl: relative over the 31 cases with |Cl| >= 0.2 (22 %), at least the mean absolute error of all 87 (0.076),
// since lift is often ~0 and a relative error means nothing there
const CD_REL_ERR = 0.086, CL_REL_ERR = 0.22, CL_ABS_ERR = 0.076;
const pm = (e) => `±${Math.max(e, 0.01).toFixed(2)}`;

function sci(x) {                                                      // 500000 -> 5·10⁵
  const e = Math.floor(Math.log10(x));
  return `${+(x / 10 ** e).toFixed(1)}·10${[...String(e)].map((c) => "⁰¹²³⁴⁵⁶⁷⁸⁹"[c]).join("")}`;
}

function updateReadout(parts) {                              // forces blended like the fields
  const m = state.current.meta, re = parts[0].f.re, sum = (key) => parts.reduce((s, p) => s + p.w * (p.f[key] ?? 0), 0);
  state.force = { cd: sum("Cd"), cl: sum("Cl"), cs: sum("Cs") };
  $("#cd").textContent = fmt(state.force.cd);
  $("#cd-pm").textContent = pm(CD_REL_ERR * Math.abs(state.force.cd));
  showLift();
  if (speedEl) {
    $("#re").textContent = `Re = ${sci(re)}`;
    speedEl.value = String(m.res.indexOf(re));
  }
  state.reShown = re;
}

// lift is along world z; rolled by phi the frame's forces turn with it. Side force is along x cross z = -y, so the
// lift becomes cos(phi) Cl - sin(phi) Cs
function showLift() {
  if (!state.force) return;
  const r = state.flowRoll * DEG, cl = Math.cos(r) * state.force.cl - Math.sin(r) * state.force.cs;
  $("#cl").textContent = fmt(cl);
  $("#cl-pm").textContent = pm(Math.max(CL_ABS_ERR, CL_REL_ERR * Math.abs(cl)));
}

// ------------------------------------------------------------------------------------------------ flow speed
// The slider is commented out in index.html for now: then Re stays at index.json's default (5·10⁵, the middle one)
// and ?re= is ignored.
function setupSpeed() {
  const m = state.current.meta;
  if (!speedEl) return;
  speedEl.max = String(m.res.length - 1);
  speedEl.disabled = m.res.length < 2;
  speedEl.title = m.res.length < 2 ? "only one flow speed is computed for this shape so far" : "";
}

speedEl?.addEventListener("input", () => {
  const m = state.current?.meta;
  if (!m) return;
  state.re = m.res[+speedEl.value];
  requestBlend();
  saveUrl();
});

// ------------------------------------------------------------------------------------------------ shapes
async function loadShape(entry) {
  if (!entry.meta) {
    entry.meta = await getJSON(`${baseOf(entry)}meta.json`);
    prepareFrames(entry.meta);
  }
  if (!entry.mesh) entry.mesh = parseMesh(await getBuffer(`${baseOf(entry)}mesh.bin`));
  return entry;
}

// ------------------------------------------------------------------------------------------------ uploads
// Local preview only (website/local/serve.py; the published site has no such server): an uploaded shape is computed
// on a GPU for every orientation, (yaw 0, pitch 0) first. Its files are under uploads/<id>/ and its meta.json grows
// while the frames come in. The body stays grey until that first frame is back, and turning waits until all are.
const LOCAL = ["localhost", "127.0.0.1"].includes(location.hostname);
const isPending = (entry) => !!entry?.upload && entry.meta?.status !== "done";
const locked = () => isPending(state.current);

function addUploads(list) {
  for (const u of list) {
    if (!state.shapes.some((s) => s.name === u.name)) state.shapes.push({ name: u.name, label: u.label, base: u.base, upload: true });
  }
  buildList(state.shapes);
  drawThumbnails(state.shapes);
  for (const s of state.shapes) if (s.upload) watchUpload(s);
}

// the dot after the name: every angle computed and not opened since; opened uploads are remembered per browser
const SEEN_KEY = "cfd-seen-uploads";
function seenUploads() {
  try { return new Set(JSON.parse(localStorage.getItem(SEEN_KEY) || "[]")); } catch { return new Set(); }
}
function markSeen(entry) {
  const seen = seenUploads();
  if (seen.has(entry.name)) return;
  seen.add(entry.name);
  try { localStorage.setItem(SEEN_KEY, JSON.stringify([...seen])); } catch { /* storage blocked: the dot comes back on reload */ }
  entry.seen = true;
}
const showDot = (entry) => entry.meta?.status === "done" && !entry.seen && !seenUploads().has(entry.name);

function markUpload(entry) {
  if (entry.meta?.status === "done" && state.current === entry) markSeen(entry);   // finished while being looked at
  const dot = document.querySelector(`.shape[data-name="${entry.name}"] .dot`);
  if (dot) dot.hidden = !showDot(entry);
}

// every upload is watched, selected or not, so its dot appears when it is done and the other shapes stay usable
async function watchUpload(entry) {
  if (entry.watching) return;
  entry.watching = true;
  if (!entry.meta) {
    try {
      await loadShape(entry);
    } catch {
      entry.watching = false;
      return;
    }
  }
  while (isPending(entry) && entry.meta?.status !== "failed") {
    await new Promise((r) => setTimeout(r, 500));
    let m;
    try {
      m = await getJSON(`${entry.base}meta.json`);
    } catch {
      continue;
    }
    const changed = m.frames.length !== entry.meta.frames.length || m.status !== entry.meta.status;
    if (changed) {
      prepareFrames(m);
      entry.meta = m;
    } else {
      entry.meta.worker = m.worker;
    }
    if (state.current !== entry) continue;
    if (changed) {
      state.lastSig = null;
      state.prefetchedAround = -1;
      requestBlend();
    }
    showUploadState(entry);
  }
  entry.watching = false;
  markUpload(entry);
  if (state.current === entry) showUploadState(entry);
}

// until the first frame: the grey body over a small flow animation (#computing); then "other angles done soon" below
// the view until every orientation is back, and turning waits for them
function showUploadState(entry) {
  const m = entry.meta, n = m.frames.length, pending = isPending(entry);
  const failed = m.status === "failed" || m.status === "interrupted";
  canvas.classList.toggle("locked", locked());
  $("#upload-progress").hidden = !pending || !n || failed;
  $("#upload-progress").textContent = "other angles done soon";
  $("#computing").hidden = !pending || n > 0 || failed;
  $("#computing-text").textContent = m.worker === "waiting" ? "waiting for a free GPU" : m.worker === "starting" ? "starting the AI model" : "computing the flow";
  if (failed) {
    return setStatus(m.status === "failed" ? `could not compute this shape${m.error ? `: ${m.error}` : ""}` : "computing was interrupted, please upload it again", true);
  }
  if (!n) setStatus("");
}

function showPending(entry) {                                // no frame yet: the grey body alone
  lines.geometry.setDrawRange(0, 0);
  velPlane.visible = false;
  bodyObj.body.material.uniforms.uPending.value = 1;
  for (const id of ["#cd", "#cl"]) $(id).textContent = "–";
  for (const id of ["#cd-pm", "#cl-pm"]) $(id).textContent = "";
  state.force = null;
  state.lastSig = null;
  showUploadState(entry);
  invalidate();
}

async function selectShape(name) {
  const entry = state.shapes.find((s) => s.name === name) || state.shapes[0];
  state.current = entry;
  if (entry.upload && entry.meta?.status === "done") markUpload(entry);            // opened: its dot goes
  for (const b of document.querySelectorAll(".shape")) b.setAttribute("aria-checked", String(b.dataset.name === entry.name));
  setStatus("loading the flow…");
  try {
    await loadShape(entry);
  } catch (err) {
    if (state.current === entry) setStatus("could not load the shape data", true);
    return;
  }
  if (state.current !== entry) return;
  if (bodyObj) {
    scene.remove(bodyObj.group);
    disposeBody(bodyObj);
  }
  bodyObj = makeBody(entry, STYLES[state.style]);
  scene.add(bodyObj.group);
  velMaterial.uniforms.uMax.value = entry.meta.speed_max ?? 2;
  // a new shape keeps the orientation (within its frames); an upload still computing starts level, where its first
  // frame is
  setTarget(isPending(entry) ? new THREE.Quaternion() : state.q);
  setShown(state.qTo);
  if (entry.upload) watchUpload(entry);
  canvas.classList.toggle("locked", locked());
  $("#upload-progress").hidden = true;
  $("#computing").hidden = true;
  state.drawn = false;
  state.prefetchedAround = -1;
  state.shownFrame = state.lastSig = null;
  setupSpeed();
  applyMode();
  resize();
  updateBlend();
  saveUrl();
}

function buildList(shapes) {
  const list = $("#shape-list");
  list.innerHTML = "";
  for (const s of shapes) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "shape";
    b.dataset.name = s.name;
    b.setAttribute("role", "radio");
    b.setAttribute("aria-checked", String(state.current?.name === s.name));
    let label = s.label;
    if (s.upload) {                                          // the dot stays on the line of the last word
      const dot = `<span class="dot" title="all angles computed"${showDot(s) ? "" : " hidden"}></span>`;
      const cut = s.label.lastIndexOf(" ") + 1;
      label = `${s.label.slice(0, cut)}<span class="nowrap">${s.label.slice(cut)}${dot}</span>`;
    }
    b.innerHTML = `<canvas width="96" height="96" aria-hidden="true"></canvas><span class="name">${label}</span>`;
    if (s.credit) b.title = s.credit;                       // CC BY attribution of the cow
    b.addEventListener("click", () => { track(`shape/${s.name}`); selectShape(s.name); });
    list.append(b);
  }
  markMore();
}

// the shape list scrolls when the column is short; its bottom fades while more shapes are below
const shapeList = $("#shape-list");
const markMore = () => shapeList.classList.toggle("more", shapeList.scrollTop + shapeList.clientHeight < shapeList.scrollHeight - 2);
shapeList.addEventListener("scroll", markMore, { passive: true });
new ResizeObserver(markMore).observe(shapeList);

// one small offscreen renderer draws every thumbnail with the body material of the current style
let thumbRun = 0;
async function drawThumbnails(shapes) {
  const run = ++thumbRun, theme = STYLES[state.style], size = 96;
  await Promise.all(shapes.map((s) => loadShape(s).catch(() => null)));
  if (run !== thumbRun) return;
  const r = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
  r.setPixelRatio(1);
  r.setSize(size, size);
  r.setClearColor(0x000000, 0);
  const cam = new THREE.PerspectiveCamera(24, 1, 0.05, 50);
  cam.up.set(0, 0, 1);
  for (const s of shapes) {
    if (!s.mesh) continue;
    const scn = new THREE.Scene();
    const obj = makeBody(s, theme);
    obj.hull.material.uniforms.uRes.value.set(size, size);
    obj.hull.material.uniforms.uWidth.value = 1.0;
    obj.body.material.uniforms.uFlat.value = 1;
    scn.add(obj.group);
    const d = (obj.radius / Math.sin(12 * DEG)) * 1.02, t = 24 * DEG, az = -128 * DEG;
    cam.position.set(d * Math.cos(t) * Math.cos(az), d * Math.cos(t) * Math.sin(az), d * Math.sin(t));
    cam.lookAt(0, 0, 0);
    r.render(scn, cam);
    const c = document.querySelector(`.shape[data-name="${s.name}"] canvas`), ctx = c.getContext("2d");
    ctx.clearRect(0, 0, c.width, c.height);
    ctx.drawImage(r.domElement, 0, 0, c.width, c.height);
    disposeBody(obj);
  }
  r.dispose();
  r.forceContextLoss();
}

function applyMode() {                                      // flow lines on / off, colour field velocity / pressure / none
  const f = state.field;
  $("#lines-toggle").setAttribute("aria-pressed", String(state.lines));
  for (const b of document.querySelectorAll(".mode [data-field]")) b.setAttribute("aria-pressed", String(b.dataset.field === f));
  if (bodyObj) bodyObj.body.material.uniforms.uMode.value = f === "pressure" ? 1 : 0;
  lines.visible = state.lines;
  lineMaterial.uniforms.uPlain.value = f ? 1 : 0;
  if (f === "velocity") requestBlend();                      // the blend also builds the velocity plane
  else {
    velPlane.visible = false;
    if (statusEl.textContent.includes("velocity")) setStatus("");
  }
  $("#legend-cp").hidden = f !== "pressure";                 // without a field, the lines are speed-coloured in engineering
  $("#legend-u").hidden = !(f === "velocity" || (!f && state.lines && state.style === "engineering"));
  invalidate();
}

function applyStyle(name) {
  state.style = name in STYLES ? name : "paper";
  const t = STYLES[state.style];
  document.documentElement.dataset.style = state.style;
  for (const b of document.querySelectorAll(".styles button")) b.setAttribute("aria-pressed", String(b.dataset.style === state.style));
  lineMaterial.uniforms.uTheme.value = t;
  velMaterial.uniforms.uTheme.value = t;
  if (bodyObj) {
    bodyObj.body.material.uniforms.uTheme.value = t;
    bodyObj.hull.material.uniforms.uColor.value.copy(OUTLINE[t]);
  }
  try { localStorage.setItem("cfd-style", state.style); } catch { /* storage blocked */ }
  applyMode();
  if (state.shapes.length) drawThumbnails(state.shapes);
}

let urlTimer = 0;
function saveUrl() {
  clearTimeout(urlTimer);
  urlTimer = setTimeout(() => {
    if (!state.current?.meta) return;
    const o = state.oriTo, roll = Math.round(angleDiff(o.roll, 0));
    const p = new URLSearchParams({ shape: state.current.name, yaw: String(Math.round(norm360(o.yaw)) % 360) });
    if (Math.round(o.pitch)) p.set("pitch", String(Math.round(o.pitch)));
    if (roll) p.set("roll", String(roll));
    if (state.reShown !== state.current.meta.re) p.set("re", String(state.reShown));
    if (state.field) p.set("mode", state.field);
    if (!state.lines) p.set("lines", "0");
    if (!state.smooth) p.set("smooth", "0");
    if (Math.abs(zoomTo - 1) > 0.005) p.set("zoom", zoomTo.toFixed(2));
    if (state.style !== "paper") p.set("style", state.style);
    history.replaceState(null, "", `?${p}${location.hash}`);
  }, 300);
}

// ------------------------------------------------------------------------------------------------ interaction
let drag = null, pinch = 0;
const pointers = new Map();                                  // one pointer turns the body, two (touch) pinch-zoom
const spread = () => {
  const [a, b] = [...pointers.values()];
  return Math.hypot(a.x - b.x, a.y - b.y);
};

canvas.addEventListener("pointerdown", (ev) => {
  pointers.set(ev.pointerId, { x: ev.clientX, y: ev.clientY });
  canvas.setPointerCapture(ev.pointerId);
  if (pointers.size === 2) {
    endDrag();                                               // a second finger zooms instead of turning
    pinch = spread();
    once("zoom");
  } else if (pointers.size === 1) {
    drag = { x: ev.clientX, y: ev.clientY };
    once("rotate");
    canvas.classList.add("dragging");
  }
});
canvas.addEventListener("pointermove", (ev) => {
  if (!pointers.has(ev.pointerId)) return;
  pointers.set(ev.pointerId, { x: ev.clientX, y: ev.clientY });
  if (pinch && pointers.size === 2) {
    const s = spread();
    if (s > 0) zoomBy(s / pinch);
    pinch = s;
    return;
  }
  if (!drag || !state.current?.meta || locked()) return;
  const arc = arcFrame(), a = arcPoint(arc, drag.x, drag.y), b = arcPoint(arc, ev.clientX, ev.clientY);
  drag = { x: ev.clientX, y: ev.clientY };
  const axis = new THREE.Vector3().crossVectors(a, b), s = axis.length();
  if (s > 1e-9) turnBy(axis.divideScalar(s), Math.atan2(s, a.dot(b)) * arc.r * DRAG_SPEED);
});

// The drag is a virtual trackball around the body (Holroyd's sphere, a hyperbola outside it): through the middle,
// sideways turns the body about the vertical axis and up / down rolls it about the flow axis; out at the sides the
// drag twists it about the view axis, e.g. grab the tail and lift it. The axes are the flow's (x right, z up, -y
// towards the viewer), not the slightly tilted camera's, so sideways stays a pure turn of the level body.
const ARC_R = 0.5, DRAG_SPEED = 0.45;                        // sphere radius / view height; degrees per pixel at its centre

function arcFrame() {                                        // the body's centre on screen and the sphere radius (px)
  const rect = canvas.getBoundingClientRect(), c = new THREE.Vector3().project(camera);
  return { x: rect.left + ((c.x + 1) / 2) * rect.width, y: rect.top + ((1 - c.y) / 2) * rect.height, r: ARC_R * rect.height };
}

function arcPoint(arc, x, y) {
  const px = (x - arc.x) / arc.r, py = (arc.y - y) / arc.r, d2 = px * px + py * py;
  return new THREE.Vector3(px, -(d2 <= 0.5 ? Math.sqrt(1 - d2) : 0.5 / Math.sqrt(d2)), py).normalize();
}

function setTarget(q) {                                      // qTo, kept within the computed tilt range
  state.qTo.copy(q);
  let o = decompose(state.qTo, state.oriTo);
  const m = state.current?.meta;
  if (m) {
    const [p0, p1] = pitchRange(m, state.re), p = Math.min(p1, Math.max(p0, o.pitch));
    if (Math.abs(p - o.pitch) > 1e-6) {
      state.qTo.copy(frameQuat(o.yaw, p, state.qTo));
      o = decompose(state.qTo, { yaw: o.yaw, pitch: p });
    }
  }
  state.oriTo = o;
}

function setShown(q) {
  state.q.copy(q);
  state.ori = decompose(state.q, state.ori);
}

const hideHint = () => $("#hint").classList.add("gone");    // "drag to rotate…" goes with the first turn

function turnBy(axis, deg) {                                 // turn the target about a world axis
  hideHint();
  setTarget(axisQuat(axis, deg).multiply(state.qTo));
  if (state.smooth && drag) setShown(state.qTo);             // the drag moves the body directly, keys glide
  requestBlend();
  invalidate();
}

function snapTarget() {                                      // qTo onto the closest computed frame, keeping its roll
  const m = state.current?.meta;
  if (!m) return;
  const f = m.frames[nearestFrame(m, state.re, state.oriTo.yaw, state.oriTo.pitch)];
  setTarget(frameQuat(f.yaw, f.pitch, state.qTo));
}

const endDrag = () => {
  if (!drag) return;
  drag = null;
  canvas.classList.remove("dragging");
  if (!state.smooth) snapTarget();                           // the next drag starts from the frame on screen
  saveUrl();
};
const endPointer = (ev) => {
  pointers.delete(ev.pointerId);
  if (pointers.size < 2) pinch = 0;
  endDrag();                                                 // after a pinch, turning needs a fresh touch
};
canvas.addEventListener("pointerup", endPointer);
canvas.addEventListener("pointercancel", endPointer);
canvas.addEventListener("wheel", (ev) => {                   // mouse wheel; a trackpad pinch arrives as ctrl + wheel
  ev.preventDefault();
  once("zoom");
  const px = ev.deltaMode === 1 ? 16 : ev.deltaMode === 2 ? 400 : 1;
  zoomBy(Math.exp(-ev.deltaY * px * (ev.ctrlKey ? 0.01 : 0.0015)));
}, { passive: false });
canvas.addEventListener("dblclick", () => { zoomTo = 1; saveUrl(); });
// arrow keys turn like the drag through the middle: left / right about the vertical axis, up / down roll about the
// flow axis. Smooth turning: glide 10 degrees; else left / right go on to the next computed frame and the roll (free
// anyway) steps 15 degrees.
canvas.addEventListener("keydown", (ev) => {
  if (["+", "=", "-", "0"].includes(ev.key)) {               // + / - zoom, 0 resets
    ev.preventDefault();
    if (ev.key === "0") {
      zoomTo = 1;
      saveUrl();
    } else {
      zoomBy(ev.key === "-" ? 1 / 1.25 : 1.25);
    }
    return;
  }
  const m = state.current?.meta;
  const side = ev.key === "ArrowLeft" ? -1 : ev.key === "ArrowRight" ? 1 : 0;
  const up = ev.key === "ArrowUp" ? 1 : ev.key === "ArrowDown" ? -1 : 0;
  if (!m || (!side && !up) || locked()) return;
  hideHint();
  ev.preventDefault();
  if (state.smooth || up) {
    turnBy(side ? Z_AXIS : X_AXIS, side ? 10 * side : -(state.smooth ? 10 : 15) * up);
  } else {
    snapTarget();
    const k0 = nearestFrame(m, state.re, state.oriTo.yaw, state.oriTo.pitch), start = state.qTo.clone();
    for (let deg = 10; deg <= 180; deg += 10) {              // the first turn that reaches another frame
      setTarget(axisQuat(Z_AXIS, deg * side).multiply(start));
      if (nearestFrame(m, state.re, state.oriTo.yaw, state.oriTo.pitch) !== k0) break;
    }
    snapTarget();
    requestBlend();
  }
  saveUrl();
});

function applySmooth() {
  $("#smooth-toggle").setAttribute("aria-pressed", String(state.smooth));
  $("#smooth-hint").hidden = !state.smooth;
}

$("#smooth-toggle").addEventListener("click", () => {
  state.smooth = !state.smooth;
  track(state.smooth ? "view/smooth-on" : "view/smooth-off");
  if (state.smooth) {                                        // continue from the body on screen
    setTarget(state.q);
  } else {
    snapTarget();                                            // back to the closest computed frame
  }
  applySmooth();
  requestBlend();
  saveUrl();
});
$("#lines-toggle").addEventListener("click", () => {
  state.lines = !state.lines;
  track(state.lines ? "view/lines-on" : "view/lines-off");
  applyMode();
  saveUrl();
});
for (const b of document.querySelectorAll(".mode [data-field]")) {      // pick a field; pick it again to hide it
  b.addEventListener("click", () => {
    state.field = state.field === b.dataset.field ? null : b.dataset.field;
    if (state.field) track(`view/${state.field}`);
    applyMode();
    saveUrl();
  });
}
for (const b of document.querySelectorAll(".styles button")) {
  b.addEventListener("click", () => { applyStyle(b.dataset.style); saveUrl(); });
}

// ------------------------------------------------------------------------------------------------ render loop
let visible = true, last = performance.now(), lastDraw = 0;
new IntersectionObserver(([e]) => { visible = e.isIntersecting; }).observe(stage);
new ResizeObserver(resize).observe(stage);

// Draws only when something changed, while the body turns, or at ~30 fps for the travelling dashes.
function tick(now) {
  requestAnimationFrame(tick);
  const dt = Math.min(0.1, (now - last) / 1000);
  last = now;
  if (!visible || document.hidden || bodyObj?.entry !== state.current) return;   // also skips while a shape loads
  // smooth turning: arrow-key steps glide and the flow blends along; else the body turns to the frame on screen
  const sf = state.shownFrame, fixed = !state.smooth && sf;
  const goal = fixed ? frameQuat(sf.yaw, sf.pitch, state.qTo) : state.qTo;
  const turning = state.q.angleTo(goal) > 2e-4;
  if (turning) {
    const k = reducedMotion ? 1 : Math.min(1, dt * (state.smooth ? 10 : 16));
    setShown(state.q.clone().slerp(goal, k));
    if (state.smooth) requestBlend();
  }
  if (blendDirty) updateBlend();
  // the lines and the plane: the shown frame rolled to match the body (smooth turning: exactly the body's roll)
  const roll = fixed ? bestRoll(quatMat(state.q), rotation(sf.yaw, sf.pitch)) : state.ori.roll;
  if (roll !== state.flowRoll) {
    state.flowRoll = roll;
    flowGroup.rotation.x = roll * DEG;
    if (velPlane.visible && velParts) rollVel();
    showLift();
    invalidate();
  }
  if (zoom !== zoomTo) {                                     // zoom eases towards its target
    zoom = reducedMotion || Math.abs(zoomTo - zoom) < 1e-3 ? zoomTo : zoom + (zoomTo - zoom) * Math.min(1, dt * 12);
    placeCamera();
  }
  // the dashes travel with the flow speed: U_inf grows with Re (same body, same fluid), shown damped
  if (!reducedMotion) dashTime += dt * 0.4 * Math.sqrt(state.reShown / state.current.meta.re);
  const dashes = lines.visible && !reducedMotion && now - lastDraw > 32;
  if (!dirty && !turning && !dashes) return;
  bodyObj.group.quaternion.copy(state.q);
  lineMaterial.uniforms.uTime.value = dashTime;
  renderer.render(scene, camera);
  dirty = false;
  lastDraw = now;
}

// ------------------------------------------------------------------------------------------------ centre windows
// ("upload your own" and its window belong to account.js)
for (const d of document.querySelectorAll("dialog")) {
  d.querySelector(".close").addEventListener("click", () => d.close());
  d.addEventListener("click", (ev) => {                      // a click on the backdrop closes it
    const r = d.getBoundingClientRect();
    if (ev.clientX < r.left || ev.clientX > r.right || ev.clientY < r.top || ev.clientY > r.bottom) d.close();
  });
}
$("#how").addEventListener("click", () => { track("open/how"); $("#about-dialog").showModal(); });
$("#credits").addEventListener("click", () => { track("open/credits"); $("#credits-dialog").showModal(); });

// ------------------------------------------------------------------------------------------------ start
async function main() {
  applyStyle(document.documentElement.dataset.style);
  let index;
  try {
    index = await getJSON(DATA + "index.json");
  } catch (err) {
    setStatus(location.protocol === "file:" ? "open this page through a web server (python -m http.server)" : "could not load the flow data", true);
    return;
  }
  state.shapes = index.shapes.map((s) => ({ ...s }));
  state.re = state.reShown = index.re || state.re;
  if (LOCAL) {                                               // uploads of the local preview server, if it runs
    try {
      const r = await fetch("api/local", { cache: "no-store" });
      if (r.ok) state.shapes.push(...(await r.json()).uploads.map((u) => ({ name: u.name, label: u.label, base: u.base, upload: true })));
    } catch { /* plain http.server: no uploads */ }
  }
  window.addEventListener("cfd:upload", (ev) => {           // account.js, after the local server took a file
    addUploads([ev.detail]);
    selectShape(ev.detail.name);
  });
  buildList(state.shapes);
  for (const s of state.shapes) if (s.upload) watchUpload(s);
  const p = new URLSearchParams(location.search);
  const pitch = Math.min(90, Math.max(-90, Number(p.get("pitch")) || 0));
  setTarget(orientQuat(Number(p.get("yaw")) || 0, pitch, Number(p.get("roll")) || 0));
  setShown(state.qTo);
  if (speedEl && Number(p.get("re")) > 0) state.re = Number(p.get("re"));
  if (FIELDS.includes(p.get("mode"))) state.field = p.get("mode");
  if (p.get("lines") === "0") state.lines = false;
  state.smooth = p.get("smooth") !== "0";
  zoom = zoomTo = clampZoom(Number(p.get("zoom")) || 1);
  applySmooth();
  resize();
  requestAnimationFrame(tick);
  await selectShape(p.get("shape") || index.default);
  drawThumbnails(state.shapes);
}

main();

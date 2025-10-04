const H = 256, W = 256;
const MEAN = [0.485, 0.456, 0.406];
const STD  = [0.229, 0.224, 0.225];

const epEl   = document.getElementById('ep');
const latEl  = document.getElementById('lat');
const thrEl  = document.getElementById('thr');
const fileEl = document.getElementById('file');
const preview= document.getElementById('preview');
const canvas = document.getElementById('overlay');
const ctx    = canvas.getContext('2d');
const saveBtn= document.getElementById('save');

let session = null;
let lastProb = null;

// ---------- HARDEN ORT INIT ----------
(async () => {
  try {
    // Ensure ORT is loaded
    console.log('ORT version:', (typeof ort !== 'undefined') ? ort.version : 'NOT LOADED');
    if (typeof ort === 'undefined') throw new Error('ort.min.js did not load');

ort.env.wasm.wasmPaths = '/';   // absolute: files live at site root
ort.env.wasm.numThreads = 1;    // avoid SharedArrayBuffer requirements
ort.env.wasm.simd = false;
ort.env.wasm.proxy = false;     // no worker proxy
const MODEL = '/lesion_256.onnx';


    // Probe model URL (bypass any stale cache)
    const probe = await fetch(`${MODEL}?v=8`, { cache: 'reload' });
    console.log('Model fetch:', probe.status, probe.headers.get('content-type'));
    if (!probe.ok) throw new Error(`Model fetch failed: ${probe.status}`);

    // Prefer WASM first (most robust), then try GPU options
    const tryEPs = [
      ['wasm'],
      ['webgpu'],
      ['webgl']
    ];

    let lastErr = null;
    for (const providers of tryEPs) {
      try {
        console.log('Trying EP:', providers[0]);
        session = await ort.InferenceSession.create(`${MODEL}?v=8`, { executionProviders: providers });
        epEl.textContent = providers[0];
        console.log('Initialized with', providers[0]);
        lastErr = null;
        break;
      } catch (e) {
        console.error('EP failed:', providers[0], e);
        lastErr = e;
      }
    }

    if (!session) throw lastErr || new Error('No backend initialized');
  } catch (e) {
    console.error('Init error:', e);
    alert('Failed to initialize ONNX Runtime Web.');
  }
})();
// ---------- END INIT ----------

fileEl.onchange = async () => {
  const f = fileEl.files?.[0];
  if (!f || !session) return;

  preview.src = URL.createObjectURL(f);
  await new Promise(r => preview.onload = r);

  const tmp = document.createElement('canvas');
  tmp.width = W; tmp.height = H;
  const tctx = tmp.getContext('2d', { willReadFrequently: true });
  tctx.drawImage(preview, 0, 0, W, H);

  const rgba = tctx.getImageData(0, 0, W, H).data;
  const chw = new Float32Array(1 * 3 * H * W);
  let oR = 0, oG = H*W, oB = 2*H*W;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = (y*W + x) * 4;
      const r = rgba[i] / 255, g = rgba[i+1] / 255, b = rgba[i+2] / 255;
      chw[oR++] = (r - MEAN[0]) / STD[0];
      chw[oG++] = (g - MEAN[1]) / STD[1];
      chw[oB++] = (b - MEAN[2]) / STD[2];
    }
  }

  const feeds = { input: new ort.Tensor('float32', chw, [1,3,H,W]) };

  await session.run(feeds); // warm
  const t0 = performance.now();
  const out = await session.run(feeds);
  const t1 = performance.now();
  latEl.textContent = Math.round(t1 - t0);

  const logits = out.logits.data;
  const N = H * W;
  lastProb = new Float32Array(N);
  for (let i = 0; i < N; i++) lastProb[i] = 1 / (1 + Math.exp(-logits[i]));
  renderOverlay();
};

thrEl.oninput = renderOverlay;

function renderOverlay(){
  if (!lastProb) return;
  const thr = parseFloat(thrEl.value);
  const im = ctx.createImageData(W, H);
  for (let i = 0; i < H*W; i++) {
    const m = lastProb[i] > thr ? 255 : 0;
    im.data[i*4+0] = 255;
    im.data[i*4+1] = 0;
    im.data[i*4+2] = 0;
    im.data[i*4+3] = Math.round(m * 0.5);
  }
  ctx.putImageData(im, 0, 0);
}

saveBtn.onclick = () => {
  if (!lastProb) return;
  const im = ctx.createImageData(W, H);
  for (let i = 0; i < H*W; i++) {
    const v = lastProb[i] > parseFloat(thrEl.value) ? 255 : 0;
    im.data[i*4+0] = v; im.data[i*4+1] = v; im.data[i*4+2] = v; im.data[i*4+3] = 255;
  }
  ctx.putImageData(im, 0, 0);
  const a = document.createElement('a');
  a.download = 'mask_256.png';
  a.href = canvas.toDataURL('image/png');
  a.click();
  renderOverlay();
};

const H = 256, W = 256;
// ImageNet normalization
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

(async () => {
  // Prefer WebGPU → WebGL → WASM
  const backends = [['webgpu'], ['webgl'], ['wasm']];
  for (const providers of backends) {
    try {
      session = await ort.InferenceSession.create('lesion_256_fp16.onnx', { executionProviders: providers });
      epEl.textContent = providers[0];
      break;
    } catch (e) { /* try next */ }
  }
  if (!session) { alert('Failed to initialize ONNX Runtime Web.'); }
})();

fileEl.onchange = async () => {
  const f = fileEl.files?.[0];
  if (!f) return;

  // show preview
  preview.src = URL.createObjectURL(f);
  await new Promise(r => preview.onload = r);

  // draw to temp canvas at 256×256
  const tmp = document.createElement('canvas');
  tmp.width = W; tmp.height = H;
  const tctx = tmp.getContext('2d', { willReadFrequently: true });
  tctx.drawImage(preview, 0, 0, W, H);

  // HWC uint8 → NCHW float32 with mean/std
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

  // warmup
  await session.run(feeds);

  // timed run
  const t0 = performance.now();
  const out = await session.run(feeds);
  const t1 = performance.now();
  latEl.textContent = Math.round(t1 - t0);

  // logits → sigmoid
  const logits = out.logits.data; // flattened [1,1,H,W]
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
    // red overlay, 50% alpha
    im.data[i*4+0] = 255;
    im.data[i*4+1] = 0;
    im.data[i*4+2] = 0;
    im.data[i*4+3] = Math.round(m * 0.5); // 0..127
  }
  ctx.putImageData(im, 0, 0);
}

saveBtn.onclick = () => {
  if (!lastProb) return;
  const thr = parseFloat(thrEl.value);
  const im = ctx.createImageData(W, H);
  for (let i = 0; i < H*W; i++) {
    const v = lastProb[i] > thr ? 255 : 0;
    im.data[i*4+0] = v; im.data[i*4+1] = v; im.data[i*4+2] = v; im.data[i*4+3] = 255;
  }
  ctx.putImageData(im, 0, 0);
  const a = document.createElement('a');
  a.download = 'mask_256.png';
  a.href = canvas.toDataURL('image/png');
  a.click();
  renderOverlay(); // restore overlay view
};

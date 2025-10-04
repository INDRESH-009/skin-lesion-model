const CACHE = 'lesionseg-v11'; // bump
const ASSETS = [
  './',
  './index.html',
  './app.js',
  './manifest.webmanifest',
  './ort.min.js',
  './lesion_256.onnx',

  // legacy wasm binaries
  './ort-wasm.wasm',
  './ort-wasm-simd.wasm',
  './ort-wasm-threaded.wasm',
  './ort-wasm-threaded-simd.wasm',

  // JSEP loaders + binaries (these were 404 in your console)
  './ort-wasm.jsep.mjs',
  './ort-wasm-simd.jsep.mjs',
  './ort-wasm-threaded.jsep.mjs',
  './ort-wasm-simd-threaded.jsep.mjs',
  './ort-wasm.jsep.wasm',
  './ort-wasm-simd.jsep.wasm',
  './ort-wasm-threaded.jsep.wasm',
  './ort-wasm-simd-threaded.jsep.wasm'
];

// (keep your install/activate/fetch handlers the same)


self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(ASSETS)));
  self.skipWaiting();
});
self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
    )
  );
  self.clients.claim();
});
self.addEventListener('fetch', e => {
  if (e.request.method !== 'GET') return;
  e.respondWith(
    caches.match(e.request).then(hit =>
      hit ||
      fetch(e.request, { cache: 'reload' }).then(resp => {
        const clone = resp.clone();
        caches.open(CACHE).then(c => c.put(e.request, clone)).catch(() => {});
        return resp;
      })
    )
  );
});

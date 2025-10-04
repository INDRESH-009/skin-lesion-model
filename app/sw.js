const CACHE = 'lesionseg-v10'; // bump
const ASSETS = [
  './',
  './index.html',
  './app.js',
  './manifest.webmanifest',
  './ort.min.js',
  './lesion_256.onnx',
  './ort-wasm.wasm',
  './ort-wasm-simd.wasm',
  './ort-wasm-threaded.wasm',
  './ort-wasm-threaded-simd.wasm',
  './ort-wasm.jsep.mjs',
  './ort-wasm-simd.jsep.mjs',
  './ort-wasm-threaded.jsep.mjs',
  './ort-wasm-simd-threaded.jsep.mjs'
];


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

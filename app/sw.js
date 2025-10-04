const CACHE = 'lesionseg-v6';  // bump this number
const ASSETS = [
  './',
  './index.html',
  './app.js',
  './manifest.webmanifest',
  './ort.min.js',
  './lesion_256.onnx'  // EXACT same filename as in app.js
];

// install/activate/fetch same as before…
self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(ASSETS)));
  self.skipWaiting();
});
self.addEventListener('activate', e => {
  e.waitUntil(caches.keys().then(keys =>
    Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
  ));
  self.clients.claim();
});
self.addEventListener('fetch', e => {
  if (e.request.method !== 'GET') return;
  e.respondWith(
    caches.match(e.request).then(hit => hit ||
      fetch(e.request, { cache: 'reload' }).then(resp => {
        const clone = resp.clone();
        caches.open(CACHE).then(c => c.put(e.request, clone)).catch(()=>{});
        return resp;
      })
    )
  );
});

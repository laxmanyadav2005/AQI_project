/**
 * sw.js - Service Worker for IND-AQI Platform PWA
 *
 * Strategy:
 *  - App Shell (HTML, CSS, JS, icons) → Cache First, then Network
 *  - API calls (/api/*, /predict_lstm, /predict) → Network First, then offline fallback
 */

const CACHE_NAME = 'aqi-cache-v1';
const OFFLINE_URL = '/';

// Assets to pre-cache on install (App Shell)
const PRECACHE_URLS = [
    '/',
    '/calculator',
    '/predict',
    '/compare',
    '/static/css/style.css',
    '/static/manifest.json',
    '/static/icon-192.png',
    '/static/icon-512.png',
    // CDN resources cached on first use (not precached to avoid CSP issues)
];

// ── Install: cache App Shell ──────────────────────────────────────────────────
self.addEventListener('install', event => {
    event.waitUntil(
        caches.open(CACHE_NAME).then(cache => {
            console.log('[SW] Pre-caching App Shell');
            return cache.addAll(PRECACHE_URLS);
        }).then(() => self.skipWaiting())
    );
});

// ── Activate: clean up old caches ────────────────────────────────────────────
self.addEventListener('activate', event => {
    event.waitUntil(
        caches.keys().then(keys =>
            Promise.all(
                keys
                    .filter(key => key !== CACHE_NAME)
                    .map(key => {
                        console.log('[SW] Deleting old cache:', key);
                        return caches.delete(key);
                    })
            )
        ).then(() => self.clients.claim())
    );
});

// ── Fetch: routing strategy ───────────────────────────────────────────────────
self.addEventListener('fetch', event => {
    const { request } = event;
    const url = new URL(request.url);

    // Skip non-GET requests and cross-origin requests
    if (request.method !== 'GET' || url.origin !== location.origin) return;

    // API & prediction endpoints → Network First
    const isApiCall = url.pathname.startsWith('/api/') ||
                      url.pathname === '/predict_lstm' ||
                      url.pathname === '/predict' ||
                      url.pathname === '/calculator';

    if (isApiCall) {
        event.respondWith(
            fetch(request)
                .catch(() => caches.match(OFFLINE_URL))
        );
        return;
    }

    // App Shell → Cache First, then Network, then offline fallback
    event.respondWith(
        caches.match(request).then(cached => {
            if (cached) return cached;

            return fetch(request).then(response => {
                // Cache valid responses
                if (response && response.status === 200 && response.type === 'basic') {
                    const responseClone = response.clone();
                    caches.open(CACHE_NAME).then(cache => cache.put(request, responseClone));
                }
                return response;
            }).catch(() => caches.match(OFFLINE_URL));
        })
    );
});

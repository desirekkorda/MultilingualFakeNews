"use strict";


const CACHE_NAME =
    "fake-news-detector-pwa-v1";


const APP_SHELL = [
    "./",
    "./index.html",
    "./styles.css",
    "./app.js",
    "./manifest.webmanifest",

    "./icons/icon-192.png",
    "./icons/icon-512.png",
    "./icons/icon-192-maskable.png",
    "./icons/icon-512-maskable.png"
];


/* -------------------------------------------------------------------------- */
/* Install                                                                    */
/* -------------------------------------------------------------------------- */

self.addEventListener(
    "install",
    (event) => {

        event.waitUntil(

            caches
                .open(CACHE_NAME)
                .then(
                    (cache) => {

                        return cache.addAll(
                            APP_SHELL
                        );
                    }
                )
        );

        self.skipWaiting();
    }
);


/* -------------------------------------------------------------------------- */
/* Activate                                                                  */
/* -------------------------------------------------------------------------- */

self.addEventListener(
    "activate",
    (event) => {

        event.waitUntil(

            caches
                .keys()
                .then(
                    (cacheNames) => {

                        return Promise.all(
                            cacheNames
                                .filter(
                                    (name) =>
                                        name !== CACHE_NAME
                                )
                                .map(
                                    (name) =>
                                        caches.delete(name)
                                )
                        );
                    }
                )
        );

        self.clients.claim();
    }
);


/* -------------------------------------------------------------------------- */
/* Fetch                                                                      */
/* -------------------------------------------------------------------------- */

self.addEventListener(
    "fetch",
    (event) => {

        const request =
            event.request;


        /*
         * Only handle GET requests.
         *
         * Prediction requests are POST requests and
         * therefore pass directly to the API.
         */
        if (
            request.method !== "GET"
        ) {
            return;
        }


        const url =
            new URL(request.url);


        /*
         * Only cache our own PWA assets.
         *
         * We don't want the service worker to intercept
         * third-party APIs or external requests.
         */
        if (
            url.origin !== self.location.origin
        ) {
            return;
        }


        event.respondWith(

            caches
                .match(request)
                .then(
                    (cachedResponse) => {

                        if (
                            cachedResponse
                        ) {
                            return cachedResponse;
                        }


                        return fetch(request)
                            .then(
                                (networkResponse) => {

                                    /*
                                     * Cache a successful response.
                                     */
                                    if (
                                        networkResponse &&
                                        networkResponse.ok
                                    ) {

                                        const copy =
                                            networkResponse.clone();

                                        caches
                                            .open(CACHE_NAME)
                                            .then(
                                                (cache) => {
                                                    cache.put(
                                                        request,
                                                        copy
                                                    );
                                                }
                                            );
                                    }

                                    return networkResponse;
                                }
                            );

                    }
                )
        );
    }
);
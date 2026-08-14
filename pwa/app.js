/*
 * Multilingual Fake News Detector - PWA frontend
 */

"use strict";


/* -------------------------------------------------------------------------- */
/* Configuration                                                              */
/* -------------------------------------------------------------------------- */

/*
 * LOCAL DEVELOPMENT:
 *
 * http://127.0.0.1:8000
 *
 * PRODUCTION:
 *
 * Replace this with the public HTTPS URL of your FastAPI service.
 *
 * Example:
 *
 * https://your-api-domain.com
 */
const API_BASE_URL =
    "http://127.0.0.1:8000";


const API_PREDICT_URL =
    `${API_BASE_URL}/api/v1/predict`;


const API_HEALTH_URL =
    `${API_BASE_URL}/api/v1/health`;


/* -------------------------------------------------------------------------- */
/* DOM references                                                              */
/* -------------------------------------------------------------------------- */

const newsText =
    document.getElementById("newsText");

const analyzeButton =
    document.getElementById("analyzeButton");

const analyzeText =
    document.getElementById("analyzeText");

const loadingSpinner =
    document.getElementById("loadingSpinner");

const clearButton =
    document.getElementById("clearButton");

const characterCount =
    document.getElementById("characterCount");

const errorMessage =
    document.getElementById("errorMessage");

const resultSection =
    document.getElementById("resultSection");

const predictionTitle =
    document.getElementById("predictionTitle");

const predictionMessage =
    document.getElementById("predictionMessage");

const confidenceBadge =
    document.getElementById("confidenceBadge");

const confidenceValue =
    document.getElementById("confidenceValue");

const inferenceTime =
    document.getElementById("inferenceTime");

const legitPercentage =
    document.getElementById("legitPercentage");

const fakePercentage =
    document.getElementById("fakePercentage");

const legitProgress =
    document.getElementById("legitProgress");

const fakeProgress =
    document.getElementById("fakeProgress");

const connectionStatus =
    document.getElementById("connectionStatus");

const installButton =
    document.getElementById("installButton");

const installHelp =
    document.getElementById("installHelp");

const closeInstallHelp =
    document.getElementById("closeInstallHelp");


/* -------------------------------------------------------------------------- */
/* Application state                                                          */
/* -------------------------------------------------------------------------- */

let deferredInstallPrompt = null;


/* -------------------------------------------------------------------------- */
/* Character counter                                                          */
/* -------------------------------------------------------------------------- */

function updateCharacterCount() {

    const count =
        newsText.value.length;

    characterCount.textContent =
        `${count.toLocaleString()} characters`;
}


newsText.addEventListener(
    "input",
    updateCharacterCount
);


/* -------------------------------------------------------------------------- */
/* Error handling                                                             */
/* -------------------------------------------------------------------------- */

function showError(message) {

    errorMessage.textContent =
        message;

    errorMessage.classList.remove(
        "hidden"
    );
}


function clearError() {

    errorMessage.textContent = "";

    errorMessage.classList.add(
        "hidden"
    );
}


/* -------------------------------------------------------------------------- */
/* Loading state                                                               */
/* -------------------------------------------------------------------------- */

function setLoading(isLoading) {

    analyzeButton.disabled =
        isLoading;

    if (isLoading) {

        analyzeText.textContent =
            "Analyzing...";

        loadingSpinner.classList.remove(
            "hidden"
        );

    } else {

        analyzeText.textContent =
            "Analyze Article";

        loadingSpinner.classList.add(
            "hidden"
        );
    }
}


/* -------------------------------------------------------------------------- */
/* API health                                                                  */
/* -------------------------------------------------------------------------- */

async function checkApiHealth() {

    try {

        const response =
            await fetch(
                API_HEALTH_URL,
                {
                    method: "GET",
                    headers: {
                        "Accept":
                            "application/json"
                    }
                }
            );

        if (!response.ok) {
            throw new Error(
                `API returned ${response.status}`
            );
        }

        connectionStatus.textContent =
            "API Online";

        connectionStatus.className =
            "status status-online";

    } catch (error) {

        console.error(
            "API health check failed:",
            error
        );

        connectionStatus.textContent =
            "API Unavailable";

        connectionStatus.className =
            "status status-offline";
    }
}


/* -------------------------------------------------------------------------- */
/* Prediction                                                                  */
/* -------------------------------------------------------------------------- */

async function analyzeArticle() {

    clearError();

    const text =
        newsText.value.trim();

    if (!text) {

        showError(
            "Please enter some news text before analyzing."
        );

        newsText.focus();

        return;
    }

    setLoading(true);

    const startTime =
        performance.now();

    try {

        const response =
            await fetch(
                API_PREDICT_URL,
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json",

                        "Accept":
                            "application/json"
                    },

                    body: JSON.stringify({
                        text: text
                    })
                }
            );


        if (!response.ok) {

            let detail =
                `Request failed (${response.status}).`;

            try {

                const errorData =
                    await response.json();

                if (errorData.detail) {
                    detail =
                        errorData.detail;
                }

            } catch (_) {
                // Keep the default error.
            }

            throw new Error(detail);
        }


        const result =
            await response.json();


        const clientElapsed =
            (
                performance.now()
                - startTime
            ) / 1000;


        displayResult(
            result,
            clientElapsed
        );


        connectionStatus.textContent =
            "API Online";

        connectionStatus.className =
            "status status-online";

    } catch (error) {

        console.error(
            "Prediction error:",
            error
        );

        showError(
            "Unable to analyze the article. "
            + "Please check your connection and try again."
        );

        connectionStatus.textContent =
            "API Error";

        connectionStatus.className =
            "status status-offline";

    } finally {

        setLoading(false);
    }
}


/* -------------------------------------------------------------------------- */
/* Display prediction                                                         */
/* -------------------------------------------------------------------------- */

function displayResult(
    result,
    clientElapsed
) {

    const prediction =
        result.prediction;


    const confidence =
        Number(
            result.confidence ?? 0
        );


    const probabilities =
        result.probabilities ?? {};


    const legitProbability =
        Number(
            probabilities.Legit ?? 0
        );


    const fakeProbability =
        Number(
            probabilities.Fake ?? 0
        );


    predictionTitle.textContent =
        prediction === "Legit"
            ? "Likely Legitimate News"
            : "Likely Fake News";


    if (
        prediction ===
        "Legit"
    ) {

        predictionMessage.textContent =
            "The article appears consistent "
            + "with patterns learned from "
            + "legitimate news articles.";

        predictionMessage.className =
            "prediction-message legit";

    } else {

        predictionMessage.textContent =
            "The article contains patterns "
            + "commonly associated with "
            + "misinformation.";

        predictionMessage.className =
            "prediction-message fake";
    }


    confidenceBadge.textContent =
        `Confidence ${(
            confidence * 100
        ).toFixed(2)}%`;


    confidenceValue.textContent =
        `${(
            confidence * 100
        ).toFixed(2)}%`;


    legitPercentage.textContent =
        `${(
            legitProbability * 100
        ).toFixed(2)}%`;


    fakePercentage.textContent =
        `${(
            fakeProbability * 100
        ).toFixed(2)}%`;


    requestAnimationFrame(
        () => {

            legitProgress.style.width =
                `${(
                    legitProbability * 100
                )}%`;

            fakeProgress.style.width =
                `${(
                    fakeProbability * 100
                )}%`;
        }
    );


    /*
     * Show client-observed timing in the PWA.
     *
     * This includes browser/API network overhead.
     * It is intentionally different from the model-level
     * inference time measured during offline benchmarking.
     */
    inferenceTime.textContent =
        `${clientElapsed.toFixed(2)} sec`;


    resultSection.classList.remove(
        "hidden"
    );


    resultSection.scrollIntoView({
        behavior: "smooth",
        block: "start"
    });
}


/* -------------------------------------------------------------------------- */
/* Clear                                                                      */
/* -------------------------------------------------------------------------- */

clearButton.addEventListener(
    "click",
    () => {

        newsText.value = "";

        updateCharacterCount();

        clearError();

        resultSection.classList.add(
            "hidden"
        );

        newsText.focus();
    }
);


/* -------------------------------------------------------------------------- */
/* Analyze button                                                             */
/* -------------------------------------------------------------------------- */

analyzeButton.addEventListener(
    "click",
    analyzeArticle
);


/* -------------------------------------------------------------------------- */
/* Keyboard shortcut                                                          */
/* -------------------------------------------------------------------------- */

newsText.addEventListener(
    "keydown",
    (event) => {

        /*
         * Ctrl/Cmd + Enter
         */
        if (
            (event.ctrlKey || event.metaKey)
            &&
            event.key === "Enter"
        ) {

            event.preventDefault();

            analyzeArticle();
        }
    }
);


/* -------------------------------------------------------------------------- */
/* PWA installation                                                           */
/* -------------------------------------------------------------------------- */

window.addEventListener(
    "beforeinstallprompt",
    (event) => {

        event.preventDefault();

        deferredInstallPrompt =
            event;

        installButton.classList.remove(
            "hidden"
        );
    }
);


installButton.addEventListener(
    "click",
    async () => {

        if (
            deferredInstallPrompt
        ) {

            deferredInstallPrompt.prompt();

            const choice =
                await deferredInstallPrompt.userChoice;

            console.log(
                "Install choice:",
                choice.outcome
            );

            deferredInstallPrompt = null;

            installButton.classList.add(
                "hidden"
            );

            return;
        }

        /*
         * iOS/Safari and browsers that don't
         * expose beforeinstallprompt.
         */
        installHelp.classList.remove(
            "hidden"
        );
    }
);


closeInstallHelp.addEventListener(
    "click",
    () => {

        installHelp.classList.add(
            "hidden"
        );
    }
);


installHelp.addEventListener(
    "click",
    (event) => {

        if (
            event.target ===
            installHelp
        ) {

            installHelp.classList.add(
                "hidden"
            );
        }
    }
);


/* -------------------------------------------------------------------------- */
/* Service worker                                                             */
/* -------------------------------------------------------------------------- */

if (
    "serviceWorker"
    in navigator
) {

    window.addEventListener(
        "load",
        async () => {

            try {

                const registration =
                    await navigator.serviceWorker.register(
                        "./service-worker.js"
                    );

                console.log(
                    "Service worker registered:",
                    registration.scope
                );

            } catch (error) {

                console.error(
                    "Service worker registration failed:",
                    error
                );
            }
        }
    );
}


/* -------------------------------------------------------------------------- */
/* Initial startup                                                             */
/* -------------------------------------------------------------------------- */

updateCharacterCount();

checkApiHealth();
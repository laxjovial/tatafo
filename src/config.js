let loaded_fastapi_base_url;

try {
    // This global variable would be injected by the backend when serving index.html
    if (typeof __fastapi_base_url__ !== 'undefined' && __fastapi_base_url__) {
        loaded_fastapi_base_url = __fastapi_base_url__;
    } else {
        // Fallback for local development or when the variable is not injected
        console.warn("Using fallback FastAPI base URL. __fastapi_base_url__ was not provided by the environment.");
        loaded_fastapi_base_url = "https://laughing-chainsaw-r4w9vppqxx563grx-8000.app.github.dev";
    }
} catch (e) {
    console.error("Error loading FastAPI base URL:", e);
    loaded_fastapi_base_url = "https://laughing-chainsaw-r4w9vppqxx563grx-8000.app.github.dev";
}

export const FASTAPI_BASE_URL = loaded_fastapi_base_url;

import { initializeApp } from 'firebase/app';
import { getAuth } from 'firebase/auth';
import { getFirestore } from 'firebase/firestore';

// Fallback configuration if the one from the environment is not available
const fallbackConfig = {
    apiKey: "AIzaSyAmU1jL_OumyVv0XwE8yWnVl3UJuONCZ3E",
    authDomain: "tatafo-assistant.firebaseapp.com",
    projectId: "tatafo-assistant",
    storageBucket: "tatafo-assistant.appspot.com",
    messagingSenderId: "308839803512",
    appId: "1:308839803512:web:e2a9328c49c50832f5db8d"
};

let firebaseConfig = {};

try {
    if (typeof __firebase_config !== 'undefined' && __firebase_config) {
        firebaseConfig = JSON.parse(__firebase_config);
    } else {
        console.warn("Using fallback Firebase config. __firebase_config was not provided by the environment.");
        firebaseConfig = fallbackConfig;
    }
} catch (e) {
    console.error("Error parsing __firebase_config:", e);
    firebaseConfig = fallbackConfig;
}

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
const db = getFirestore(app);

export { app, auth, db };

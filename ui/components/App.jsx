import React, { useState, useEffect, useCallback } from 'react';
import { initializeApp } from 'firebase/app';
import { getAuth, signInAnonymously, signInWithCustomToken, onAuthStateChanged } from 'firebase/auth';
import { getFirestore } from 'firebase/firestore'; // Only need getFirestore here, other imports are in AnalyticsDashboard
import { initializeAnalytics, logEvent } from './utils/analytics_tracker'; // Ensure correct path

// Import the components
import AnalyticsDashboard from './AnalyticsDashboard';
import UserProfile from './UserProfile'; // Import UserProfile

// Ensure __app_id and __firebase_config are available in the environment
const appId = typeof __app_id !== 'undefined' ? __app_id : 'default-app-id';
const firebaseConfig = typeof __firebase_config !== 'undefined' ? JSON.parse(__firebase_config) : {};

// Initialize Firebase outside the component to prevent re-initialization
let app, db, auth;
if (Object.keys(firebaseConfig).length > 0) {
    try {
        app = initializeApp(firebaseConfig);
        db = getFirestore(app);
        auth = getAuth(app);
        console.log("Firebase initialized successfully in App.jsx");
    } catch (error) {
        console.error("Error initializing Firebase in App.jsx:", error);
    }
} else {
    console.warn("Firebase config not found. Application may not function correctly.");
}

const App = () => {
    const [currentPage, setCurrentPage] = useState('home');
    const [currentUserId, setCurrentUserId] = useState(null);
    const [isAnalyticsInitialized, setIsAnalyticsInitialized] = useState(false);

    useEffect(() => {
        if (!auth || !db) {
            console.warn("Firebase Auth or Firestore not available. Cannot initialize analytics with user context.");
            setIsAnalyticsInitialized(true); // Allow app to proceed even without analytics if Firebase isn't ready
            return;
        }

        const unsubscribe = onAuthStateChanged(auth, async (user) => {
            let userIdToUse = null;
            if (user) {
                userIdToUse = user.uid;
            } else {
                try {
                    if (typeof __initial_auth_token !== 'undefined' && __initial_auth_token) {
                        await signInWithCustomToken(auth, __initial_auth_token);
                    } else {
                        await signInAnonymously(auth);
                    }
                    userIdToUse = auth.currentUser?.uid || crypto.randomUUID();
                } catch (anonError) {
                    console.error("Error signing in anonymously for analytics:", anonError);
                    userIdToUse = 'anonymous_error';
                }
            }
            setCurrentUserId(userIdToUse);

            if (db && userIdToUse) {
                initializeAnalytics(db, auth, appId, userIdToUse);
                setIsAnalyticsInitialized(true);
                // Log initial page view after auth is ready
                await logEvent('page_view', { page_name: currentPage, user_id: userIdToUse });
                console.log(`Analytics: Logged initial page_view for '${currentPage}' by user '${userIdToUse}'`);
            } else {
                console.warn("Firestore or userId not available for analytics initialization.");
                setIsAnalyticsInitialized(true); // Proceed anyway
            }
        });

        return () => unsubscribe();
    }, []);

    // Log page views when currentPage changes (after analytics is initialized)
    useEffect(() => {
        if (isAnalyticsInitialized && currentUserId) {
            (async () => {
                await logEvent('page_view', { page_name: currentPage, user_id: currentUserId });
                console.log(`Analytics: Logged page_view for '${currentPage}' by user '${currentUserId}'`);
            })();
        }
    }, [currentPage, isAnalyticsInitialized, currentUserId]);

    const renderPage = useCallback(() => {
        if (!isAnalyticsInitialized) {
            return <div className="text-center py-20 text-xl text-indigo-600">Initializing application...</div>;
        }

        switch (currentPage) {
            case 'home':
                return (
                    <div className="text-center py-20">
                        <h1 className="text-5xl font-extrabold text-indigo-800 mb-6 animate-fade-in-down">
                            Welcome to Intelli-Agent!
                        </h1>
                        <p className="text-xl text-gray-700 mb-8 animate-fade-in-up">
                            Your smart assistant for everything.
                        </p>
                        <div className="flex justify-center space-x-4">
                            <button
                                onClick={() => setCurrentPage('profile')}
                                className="bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-3 px-8 rounded-full shadow-lg transform hover:scale-105 transition duration-300"
                            >
                                Get Started
                            </button>
                            <button
                                onClick={() => setCurrentPage('analytics')}
                                className="bg-gray-200 hover:bg-gray-300 text-indigo-800 font-bold py-3 px-8 rounded-full shadow-lg transform hover:scale-105 transition duration-300"
                            >
                                View Analytics
                            </button>
                        </div>
                    </div>
                );
            case 'analytics':
                return <AnalyticsDashboard />;
            case 'profile':
                return <UserProfile userId={currentUserId} />; // Pass userId to UserProfile
            default:
                return (
                    <div className="text-center py-20 text-red-500">
                        <h1 className="text-4xl font-bold mb-4">Page Not Found</h1>
                        <p className="text-lg">The page you are looking for does not exist.</p>
                        <button
                            onClick={() => setCurrentPage('home')}
                            className="mt-8 bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-2 px-6 rounded-md shadow-lg transition duration-300"
                        >
                            Go to Home
                        </button>
                    </div>
                );
        }
    }, [currentPage, isAnalyticsInitialized, currentUserId]);

    return (
        <div className="min-h-screen bg-gradient-to-br from-indigo-50 to-purple-100 font-inter">
            <header className="bg-indigo-800 text-white p-4 shadow-md">
                <div className="container mx-auto flex justify-between items-center">
                    <h1 className="text-3xl font-bold">Intelli-Agent</h1>
                    <nav>
                        <ul className="flex justify-center space-x-6">
                            <li>
                                <button
                                    onClick={() => setCurrentPage('home')}
                                    className="text-white hover:text-indigo-200 text-lg font-medium transition duration-300"
                                >
                                    Home
                                </button>
                            </li>
                            <li>
                                <button
                                    onClick={() => setCurrentPage('analytics')}
                                    className="text-white hover:text-indigo-200 text-lg font-medium transition duration-300"
                                >
                                    Analytics
                                </button>
                            </li>
                            <li>
                                <button
                                    onClick={() => setCurrentPage('profile')}
                                    className="text-white hover:text-indigo-200 text-lg font-medium transition duration-300"
                                >
                                    User Profile
                                </button>
                            </li>
                        </ul>
                    </nav>
                </div>
            </header>
            <main className="container mx-auto p-4">
                {renderPage()}
            </main>
        </div>
    );
};

export default App;

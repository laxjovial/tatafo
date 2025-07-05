import React, { useState, useEffect } from 'react';
import { initializeApp } from 'firebase/app';
import { getAuth, signInAnonymously, signInWithCustomToken, onAuthStateChanged } from 'firebase/auth';
import { getFirestore } from 'firebase/firestore';
import { logEvent, initializeAnalytics } from '../utils/analytics_tracker.js'; // Adjusted path with .js extension

// Import your page components
import AnalyticsDashboard from './components/AnalyticsDashboard.jsx'; // Adjusted path with .jsx extension
import UserProfile from './components/UserProfile.jsx'; // Adjusted path with .jsx extension
// import OtherPage from './pages/OtherPage'; // Example other page

// Ensure __app_id and __firebase_config are available in the environment
const appId = typeof __app_id !== 'undefined' ? __app_id : 'default-app-id';
const firebaseConfig = typeof __firebase_config !== 'undefined' ? JSON.parse(__firebase_config) : {};

// Initialize Firebase outside the component to prevent re-initialization issues
let app, db, auth;
if (Object.keys(firebaseConfig).length > 0) {
    app = initializeApp(firebaseConfig);
    db = getFirestore(app);
    auth = getAuth(app);
} else {
    console.warn("Firebase config not found. Analytics will not be initialized.");
}

const App = () => {
    const [currentPage, setCurrentPage] = useState('home'); // State to simulate routing
    const [currentUserId, setCurrentUserId] = useState(null);
    const [isAnalyticsInitialized, setIsAnalyticsInitialized] = useState(false);

    // Initialize Firebase Auth and Analytics
    useEffect(() => {
        if (!auth) {
            console.warn("Firebase Auth not available. Cannot initialize analytics with user context.");
            setIsAnalyticsInitialized(true); // Allow app to proceed without user-specific analytics
            return;
        }

        const unsubscribe = onAuthStateChanged(auth, async (user) => {
            let userIdToUse = null;
            if (user) {
                userIdToUse = user.uid;
            } else {
                // Attempt anonymous sign-in if no user is authenticated
                try {
                    if (typeof __initial_auth_token !== 'undefined' && __initial_auth_token) {
                        await signInWithCustomToken(auth, __initial_auth_token);
                    } else {
                        await signInAnonymously(auth);
                    }
                    userIdToUse = auth.currentUser?.uid || crypto.randomUUID();
                } catch (anonError) {
                    console.error("Error signing in anonymously for analytics:", anonError);
                    userIdToUse = 'anonymous_error'; // Fallback for failed anonymous sign-in
                }
            }
            setCurrentUserId(userIdToUse);
            // Initialize analytics with the determined user ID
            if (db && userIdToUse) {
                initializeAnalytics(db, auth, appId, userIdToUse);
                setIsAnalyticsInitialized(true);
            } else {
                console.warn("Firestore or User ID not available for analytics initialization.");
                setIsAnalyticsInitialized(true); // Proceed even if analytics init fails
            }
        });

        return () => unsubscribe();
    }, []); // Run once on component mount

    // Log page views when currentPage changes and analytics is initialized
    useEffect(() => {
        if (isAnalyticsInitialized && currentUserId) {
            logEvent('page_view', { page_name: currentPage, user_id: currentUserId });
            console.log(`Analytics: Logged page_view for '${currentPage}' by user '${currentUserId}'`);
        }
    }, [currentPage, isAnalyticsInitialized, currentUserId]);

    const renderPage = () => {
        switch (currentPage) {
            case 'home':
                return (
                    <div className="text-center p-10">
                        <h2 className="text-3xl font-bold text-gray-800 mb-4">Welcome to the AI Assistant!</h2>
                        <p className="text-lg text-gray-600">Explore the features using the navigation below.</p>
                    </div>
                );
            case 'analytics':
                return <AnalyticsDashboard />;
            case 'profile':
                return <UserProfile userId={currentUserId} />; // Pass userId to UserProfile
            // case 'other_page':
            //     return <OtherPage />;
            default:
                return (
                    <div className="text-center p-10">
                        <h2 className="text-3xl font-bold text-gray-800 mb-4">Page Not Found</h2>
                        <p className="text-lg text-gray-600">The page you are looking for does not exist.</p>
                    </div>
                );
        }
    };

    return (
        <div className="min-h-screen bg-gray-50">
            <nav className="bg-indigo-700 p-4 shadow-md">
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
                    {/* Add other navigation buttons here */}
                </ul>
            </nav>
            <main className="container mx-auto p-4">
                {isAnalyticsInitialized ? renderPage() : (
                    <div className="text-center py-20 text-xl text-indigo-600">Initializing application...</div>
                )}
            </main>
        </div>
    );
};

export default App;

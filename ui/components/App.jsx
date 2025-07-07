import React, { useState, useEffect, useCallback } from 'react';
import { initializeApp } from 'firebase/app';
import { getAuth, signInAnonymously, signInWithCustomToken, onAuthStateChanged } from 'firebase/auth';
import { getFirestore, collection, addDoc, query, where, getDocs } from 'firebase/firestore';


// --- Analytics Tracker Functions (integrated from utils/analytics_tracker.js) ---
let dbInstance = null;
let authInstance = null;
let currentAppId = null;
let currentUserId = null;

const initializeAnalytics = (firestore_db, firebase_auth, appId, userId) => {
    dbInstance = firestore_db;
    authInstance = firebase_auth;
    currentAppId = appId;
    currentUserId = userId;
    console.log(`Analytics initialized for app_id: ${currentAppId}, user_id: ${currentUserId}`);
};

const logEvent = async (eventType, eventDetails, success = null, errorMessage = null) => {
    if (!dbInstance || !currentAppId || !currentUserId) {
        console.warn("Analytics not fully initialized. Event not logged:", eventType, eventDetails);
        return;
    }

    const eventData = {
        event_type: eventType,
        details: eventDetails,
        timestamp: new Date().toISOString(),
        user_id: currentUserId, // Use the ID from the authenticated user
        app_id: currentAppId,
    };

    if (success !== null) {
        eventData.success = success;
    }
    if (errorMessage !== null) {
        eventData.error_message = errorMessage;
    }

    try {
        // Log to a public collection for analytics
        // Corrected path to ensure an odd number of segments for a collection
        const analyticsCollectionRef = collection(dbInstance, `artifacts/${currentAppId}/public/data/analytics_events`);
        await addDoc(analyticsCollectionRef, eventData);
        console.log(`Analytics event '${eventType}' logged successfully for user ${currentUserId}.`);
    } catch (error) {
        console.error(`Error logging analytics event '${eventType}':`, error);
    }
};

// --- UserProfile Component (integrated from UserProfile.jsx) ---
// UserProfile now accepts 'auth' prop
const UserProfile = ({ userId, auth }) => {
    // IMPORTANT: Replace this with your actual Codespace URL or deployed backend URL
    const FASTAPI_BASE_URL = "https://friendly-doodle-x5x6qvv74vr6h655x-8000.app.github.dev";

    const [userData, setUserData] = useState({
        username: '',
        email: '',
        phone: '',
        address: '',
        bio: '',
        tier: '',
        roles: [],
        last_login: '',
        created_at: ''
    });
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [isEditing, setIsEditing] = useState(false);
    const [message, setMessage] = useState('');

    const fetchUserProfile = useCallback(async () => {
        if (!userId || userId === 'unauthenticated_test_user' || userId === 'anonymous_error') {
            setError("User not authenticated. Please log in to the Streamlit app first.");
            setLoading(false);
            return;
        }

        setLoading(true);
        setError(null);
        setMessage('');
        try {
            const idToken = await auth.currentUser?.getIdToken(true); // Get fresh ID token
            if (!idToken) {
                throw new Error("No authentication token available. Please log in.");
            }

            const response = await fetch(`${FASTAPI_BASE_URL}/user/profile`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${idToken}` // Use the Firebase ID token
                }
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to fetch user profile.');
            }

            const data = await response.json();
            setUserData(data.user); // Backend returns {"user": profile_data}
            await logEvent('user_profile_view', { user_id: userId, status: 'success' });
        } catch (err) {
            console.error("Error fetching user profile:", err);
            setError(err.message);
            await logEvent('user_profile_view', { user_id: userId, status: 'failure', error: err.message });
        } finally {
            setLoading(false);
        }
    }, [userId, auth, FASTAPI_BASE_URL]); // Added auth to dependencies

    useEffect(() => {
        fetchUserProfile();
    }, [fetchUserProfile]);

    const handleChange = (e) => {
        const { name, value } = e.target;
        setUserData(prevData => ({
            ...prevData,
            [name]: value
        }));
    };

    const handleSave = async () => {
        setLoading(true);
        setError(null);
        setMessage('');
        try {
            const idToken = await auth.currentUser?.getIdToken(true); // Get fresh ID token
            if (!idToken) {
                throw new Error("No authentication token available. Please log in.");
            }

            const response = await fetch(`${FASTAPI_BASE_URL}/user/profile`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${idToken}` // Use the Firebase ID token
                },
                body: JSON.stringify({
                    phone: userData.phone,
                    address: userData.address,
                    bio: userData.bio
                })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to update user profile.');
            }

            const data = await response.json();
            setMessage(data.message || 'Profile updated successfully!');
            setIsEditing(false);
            await logEvent('user_profile_update', { user_id: userId, status: 'success' });
            // Re-fetch to get the latest data, including server-side updates if any
            fetchUserProfile();
        } catch (err) {
            console.error("Error updating user profile:", err);
            setError(err.message);
            await logEvent('user_profile_update', { user_id: userId, status: 'failure', error: err.message });
        } finally {
            setLoading(false);
        }
    };

    if (loading) {
        return <div className="text-center py-20 text-xl text-indigo-600">Loading user profile...</div>;
    }

    if (error) {
        return <div className="text-center py-20 text-red-500 text-xl">Error: {error}</div>;
    }

    return (
        <div className="max-w-4xl mx-auto bg-white p-8 rounded-lg shadow-xl mt-10">
            <h2 className="text-4xl font-extrabold text-indigo-800 mb-8 text-center">User Profile</h2>

            {message && (
                <div className="bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded relative mb-6" role="alert">
                    <span className="block sm:inline">{message}</span>
                </div>
            )}

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                <div className="flex flex-col">
                    <label className="text-gray-600 text-sm font-semibold mb-1">Username:</label>
                    <input
                        type="text"
                        name="username"
                        value={userData.username}
                        readOnly
                        className="p-3 border border-gray-300 rounded-md bg-gray-100 text-gray-800 focus:outline-none"
                    />
                </div>
                <div className="flex flex-col">
                    <label className="text-gray-600 text-sm font-semibold mb-1">Email:</label>
                    <input
                        type="email"
                        name="email"
                        value={userData.email}
                        readOnly
                        className="p-3 border border-gray-300 rounded-md bg-gray-100 text-gray-800 focus:outline-none"
                    />
                </div>
                <div className="flex flex-col">
                    <label className="text-gray-600 text-sm font-semibold mb-1">Account Tier:</label>
                    <input
                        type="text"
                        name="tier"
                        value={userData.tier}
                        readOnly
                        className="p-3 border border-gray-300 rounded-md bg-gray-100 text-gray-800 focus:outline-none"
                    />
                </div>
                <div className="flex flex-col">
                    <label className="text-gray-600 text-sm font-semibold mb-1">Roles:</label>
                    <input
                        type="text"
                        name="roles"
                        value={userData.roles.join(', ')}
                        readOnly
                        className="p-3 border border-gray-300 rounded-md bg-gray-100 text-gray-800 focus:outline-none"
                    />
                </div>
                <div className="flex flex-col">
                    <label className="text-gray-600 text-sm font-semibold mb-1">Last Login:</label>
                    <input
                        type="text"
                        name="last_login"
                        value={userData.last_login ? new Date(userData.last_login).toLocaleString() : 'N/A'}
                        readOnly
                        className="p-3 border border-gray-300 rounded-md bg-gray-100 text-gray-800 focus:outline-none"
                    />
                </div>
                <div className="flex flex-col">
                    <label className="text-gray-600 text-sm font-semibold mb-1">Account Created:</label>
                    <input
                        type="text"
                        name="created_at"
                        value={userData.created_at ? new Date(userData.created_at).toLocaleString() : 'N/A'}
                        readOnly
                        className="p-3 border border-gray-300 rounded-md bg-gray-100 text-gray-800 focus:outline-none"
                    />
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                <div className="flex flex-col">
                    <label htmlFor="phone" className="text-gray-600 text-sm font-semibold mb-1">Phone:</label>
                    <input
                        type="text"
                        id="phone"
                        name="phone"
                        value={userData.phone}
                        onChange={handleChange}
                        readOnly={!isEditing}
                        className={`p-3 border rounded-md text-gray-800 focus:outline-none ${isEditing ? 'border-indigo-400' : 'border-gray-300 bg-gray-50'}`}
                    />
                </div>
                <div className="flex flex-col">
                    <label htmlFor="address" className="text-gray-600 text-sm font-semibold mb-1">Address:</label>
                    <input
                        type="text"
                        id="address"
                        name="address"
                        value={userData.address}
                        onChange={handleChange}
                        readOnly={!isEditing}
                        className={`p-3 border rounded-md text-gray-800 focus:outline-none ${isEditing ? 'border-indigo-400' : 'border-gray-300 bg-gray-50'}`}
                    />
                </div>
                <div className="col-span-1 md:col-span-2 flex flex-col">
                    <label htmlFor="bio" className="text-gray-600 text-sm font-semibold mb-1">Bio:</label>
                    <textarea
                        id="bio"
                        name="bio"
                        value={userData.bio}
                        onChange={handleChange}
                        readOnly={!isEditing}
                        rows="4"
                        className={`p-3 border rounded-md text-gray-800 focus:outline-none ${isEditing ? 'border-indigo-400' : 'border-gray-300 bg-gray-50'}`}
                    ></textarea>
                </div>
            </div>

            <div className="flex justify-center space-x-4 mt-8">
                {!isEditing ? (
                    <button
                        onClick={() => setIsEditing(true)}
                        className="bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-3 px-8 rounded-full shadow-lg transform hover:scale-105 transition duration-300"
                    >
                        Edit Profile
                    </button>
                ) : (
                    <>
                        <button
                            onClick={handleSave}
                            className="bg-green-600 hover:bg-green-700 text-white font-bold py-3 px-8 rounded-full shadow-lg transform hover:scale-105 transition duration-300"
                        >
                            Save Changes
                        </button>
                        <button
                            onClick={() => {
                                setIsEditing(false);
                                fetchUserProfile(); // Discard changes by re-fetching
                            }}
                            className="bg-red-500 hover:bg-red-600 text-white font-bold py-3 px-8 rounded-full shadow-lg transform hover:scale-105 transition duration-300"
                        >
                            Cancel
                        </button>
                    </>
                )}
            </div>
        </div>
    );
};

// --- AnalyticsDashboard Component (integrated from AnalyticsDashboard.jsx) ---
const AnalyticsDashboard = ({ db, auth, appId, currentUserId }) => {
    const [events, setEvents] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [filterEventType, setFilterEventType] = useState('');
    const [filterUserId, setFilterUserId] = useState('');
    const [filterStartDate, setFilterStartDate] = useState('');
    const [filterEndDate, setFilterEndDate] = useState('');

    const fetchAnalyticsEvents = useCallback(async () => {
        if (!db || !appId || !currentUserId) {
            setError("Firebase or User ID not available for analytics. Please ensure you are logged in.");
            setLoading(false);
            return;
        }

        setLoading(true);
        setError(null);
        try {
            // Corrected path to ensure an odd number of segments for a collection
            const analyticsCollectionRef = collection(db, `artifacts/${appId}/public/data/analytics_events`);
            let q = analyticsCollectionRef;

            if (filterEventType) {
                q = query(q, where('event_type', '==', filterEventType));
            }
            if (filterUserId) {
                q = query(q, where('user_id', '==', filterUserId));
            }

            const querySnapshot = await getDocs(q);
            let fetchedEvents = querySnapshot.docs.map(doc => ({ id: doc.id, ...doc.data() }));

            // In-memory date filtering
            if (filterStartDate) {
                const start = new Date(filterStartDate).toISOString();
                fetchedEvents = fetchedEvents.filter(event => event.timestamp >= start);
            }
            if (filterEndDate) {
                const end = new Date(filterEndDate).toISOString();
                fetchedEvents = fetchedEvents.filter(event => event.timestamp <= end);
            }

            // Sort by timestamp descending
            fetchedEvents.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));

            setEvents(fetchedEvents);
            await logEvent('analytics_dashboard_view', { user_id: currentUserId, status: 'success', filters: { eventType: filterEventType, userId: filterUserId, startDate: filterStartDate, endDate: filterEndDate } });
        } catch (err) {
            console.error("Error fetching analytics events:", err);
            setError(err.message || "Failed to fetch analytics events.");
            await logEvent('analytics_dashboard_view', { user_id: currentUserId, status: 'failure', error: err.message });
        } finally {
            setLoading(false);
        }
    }, [db, appId, currentUserId, filterEventType, filterUserId, filterStartDate, filterEndDate]);

    useEffect(() => {
        fetchAnalyticsEvents();
    }, [fetchAnalyticsEvents]);

    const handleFilterChange = (setter) => (e) => {
        setter(e.target.value);
    };

    const handleApplyFilters = () => {
        fetchAnalyticsEvents(); // Re-fetch with current filters
    };

    if (loading) {
        return <div className="text-center py-20 text-xl text-indigo-600">Loading analytics data...</div>;
    }

    if (error) {
        return <div className="text-center py-20 text-red-500 text-xl">Error: {error}</div>;
    }

    return (
        <div className="max-w-6xl mx-auto bg-white p-8 rounded-lg shadow-xl mt-10">
            <h2 className="text-4xl font-extrabold text-indigo-800 mb-8 text-center">Analytics Dashboard</h2>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
                <input
                    type="text"
                    placeholder="Filter by Event Type"
                    value={filterEventType}
                    onChange={handleFilterChange(setFilterEventType)}
                    className="p-3 border border-gray-300 rounded-md focus:outline-none focus:border-indigo-400"
                />
                <input
                    type="text"
                    placeholder="Filter by User ID"
                    value={filterUserId}
                    onChange={handleFilterChange(setFilterUserId)}
                    className="p-3 border border-gray-300 rounded-md focus:outline-none focus:border-indigo-400"
                />
                <input
                    type="date"
                    placeholder="Start Date"
                    value={filterStartDate}
                    onChange={handleFilterChange(setFilterStartDate)}
                    className="p-3 border border-gray-300 rounded-md focus:outline-none focus:border-indigo-400"
                />
                <input
                    type="date"
                    placeholder="End Date"
                    value={filterEndDate}
                    onChange={handleFilterChange(setFilterEndDate)}
                    className="p-3 border border-gray-300 rounded-md focus:outline-none focus:border-indigo-400"
                />
                <button
                    onClick={handleApplyFilters}
                    className="col-span-full md:col-span-2 lg:col-span-1 bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-3 px-6 rounded-md shadow-lg transform hover:scale-105 transition duration-300"
                >
                    Apply Filters
                </button>
            </div>

            {events.length === 0 ? (
                <p className="text-center text-gray-600 text-lg">No analytics events found for the selected filters.</p>
            ) : (
                <div className="overflow-x-auto">
                    <table className="min-w-full bg-white rounded-lg shadow-md overflow-hidden">
                        <thead className="bg-indigo-100">
                            <tr>
                                <th className="py-3 px-4 text-left text-sm font-semibold text-gray-700">Timestamp</th>
                                <th className="py-3 px-4 text-left text-sm font-semibold text-gray-700">Event Type</th>
                                <th className="py-3 px-4 text-left text-sm font-semibold text-gray-700">User ID</th>
                                <th className="py-3 px-4 text-left text-sm font-semibold text-gray-700">Details</th>
                                <th className="py-3 px-4 text-left text-sm font-semibold text-gray-700">Success</th>
                                <th className="py-3 px-4 text-left text-sm font-semibold text-gray-700">Error Message</th>
                            </tr>
                        </thead>
                        <tbody>
                            {events.map((event) => (
                                <tr key={event.id} className="border-b border-gray-200 hover:bg-gray-50">
                                    <td className="py-3 px-4 text-sm text-gray-800">{new Date(event.timestamp).toLocaleString()}</td>
                                    <td className="py-3 px-4 text-sm text-gray-800">{event.event_type}</td>
                                    <td className="py-3 px-4 text-sm text-gray-800 break-all">{event.user_id}</td>
                                    <td className="py-3 px-4 text-sm text-gray-800">
                                        <pre className="whitespace-pre-wrap text-xs bg-gray-50 p-2 rounded-md overflow-auto max-h-24">{JSON.stringify(event.details, null, 2)}</pre>
                                    </td>
                                    <td className="py-3 px-4 text-sm text-gray-800">
                                        {event.success === true && <span className="text-green-600">True</span>}
                                        {event.success === false && <span className="text-red-600">False</span>}
                                        {event.success === null && <span className="text-gray-500">N/A</span>}
                                    </td>
                                    <td className="py-3 px-4 text-sm text-gray-800">
                                        <pre className="whitespace-pre-wrap text-xs bg-gray-50 p-2 rounded-md overflow-auto max-h-24">{event.error_message || 'N/A'}</pre>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}
        </div>
    );
};


// --- Main App Component ---
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
                // Pass firebase instances and currentUserId to AnalyticsDashboard
                return <AnalyticsDashboard db={db} auth={auth} appId={appId} currentUserId={currentUserId} />;
            case 'profile':
                // Pass auth instance to UserProfile
                return <UserProfile userId={currentUserId} auth={auth} />;
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
    }, [currentPage, isAnalyticsInitialized, currentUserId, db, auth]); // Added db, auth to dependencies

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

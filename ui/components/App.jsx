import React, { useState, useEffect, useCallback } from 'react';
import { initializeApp } from 'firebase/app';
import { getAuth, signInWithEmailAndPassword, createUserWithEmailAndPassword, onAuthStateChanged, signInAnonymously, signInWithCustomToken } from 'firebase/auth';
import { getFirestore, collection, addDoc, query, where, getDocs } from 'firebase/firestore';

// --- Analytics Tracker Functions ---
// These will now use the dbInstance and authInstance passed from App component
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
        user_id: currentUserId,
        app_id: currentAppId,
    };

    if (success !== null) {
        eventData.success = success;
    }
    if (errorMessage !== null) {
        eventData.error_message = errorMessage;
    }

    try {
        const analyticsCollectionRef = collection(dbInstance, `artifacts/${currentAppId}/public/data/analytics_events`);
        await addDoc(analyticsCollectionRef, eventData);
        console.log(`Analytics event '${eventType}' logged successfully for user ${currentUserId}.`);
    } catch (error) {
        console.error(`Error logging analytics event '${eventType}':`, error);
    }
};

// --- Login Page Component ---
// Now receives 'auth' prop
const LoginPage = ({ onLoginSuccess, onNavigateToRegister, auth }) => {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    // IMPORTANT: Replace this with your actual Codespace URL or deployed backend URL.
    // ENSURE IT DOES NOT HAVE A TRAILING SLASH. Example: "https://friendly-doodle-x5x6qvv74vr6h655x-8000.app.github.dev"
    const FASTAPI_BASE_URL = "https://super-fishstick-7vwq4pp6xrjvhppgw-8000.app.github.dev"; // <--- REMOVE TRAILING SLASH IF PRESENT

    const handleLogin = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError('');
        try {
            // Ensure auth is available
            if (!auth) {
                throw new Error("Firebase Auth is not initialized.");
            }

            // 1. Call FastAPI /login endpoint to get custom token
            const response = await fetch(`${FASTAPI_BASE_URL}/login`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, password }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Login failed. Please check your credentials.');
            }

            const data = await response.json();
            const customToken = data.custom_token;
            const uid = data.uid;

            // 2. Use custom token to sign in with Firebase Client SDK
            await signInWithCustomToken(auth, customToken);

            console.log("Logged in successfully with Firebase Client SDK:", uid);
            await logEvent('user_login', { email: email, uid: uid }, true);
            onLoginSuccess(uid); // Notify parent App component of successful login

        } catch (err) {
            console.error("Login error:", err);
            setError(err.message || 'An unexpected error occurred during login.');
            await logEvent('user_login', { email: email, error: err.message }, false, err.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="flex items-center justify-center min-h-screen bg-gradient-to-br from-indigo-50 to-purple-100 p-4">
            <div className="bg-white p-8 rounded-xl shadow-2xl w-full max-w-md transform transition-all duration-300 hover:scale-[1.01]">
                <h2 className="text-4xl font-extrabold text-indigo-800 mb-6 text-center">Login</h2>
                {error && <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative mb-4 text-center">{error}</div>}
                <form onSubmit={handleLogin} className="space-y-6">
                    <div>
                        <label className="block text-gray-700 text-sm font-bold mb-2" htmlFor="email">Email</label>
                        <input
                            type="email"
                            id="email"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            className="shadow appearance-none border rounded-md w-full py-3 px-4 text-gray-700 leading-tight focus:outline-none focus:ring-2 focus:ring-indigo-500 transition duration-200"
                            placeholder="your@example.com"
                            required
                        />
                    </div>
                    <div>
                        <label className="block text-gray-700 text-sm font-bold mb-2" htmlFor="password">Password</label>
                        <input
                            type="password"
                            id="password"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            className="shadow appearance-none border rounded-md w-full py-3 px-4 text-gray-700 mb-3 leading-tight focus:outline-none focus:ring-2 focus:ring-indigo-500 transition duration-200"
                            placeholder="********"
                            required
                        />
                    </div>
                    <button
                        type="submit"
                        className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-3 px-4 rounded-md shadow-lg transform hover:scale-105 transition duration-300 flex items-center justify-center"
                        disabled={loading}
                    >
                        {loading ? (
                            <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                            </svg>
                        ) : 'Login'}
                    </button>
                </form>
                <div className="mt-6 text-center">
                    <p className="text-gray-600">Don't have an account?</p>
                    <button
                        onClick={onNavigateToRegister}
                        className="text-indigo-600 hover:text-indigo-800 font-semibold mt-2 transition duration-300"
                    >
                        Register here
                    </button>
                </div>
            </div>
        </div>
    );
};

// --- Register Page Component ---
// Now receives 'auth' prop
const RegisterPage = ({ onRegisterSuccess, onNavigateToLogin, auth }) => {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [username, setUsername] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [message, setMessage] = useState('');

    // IMPORTANT: Replace this with your actual Codespace URL or deployed backend URL.
    // ENSURE IT DOES NOT HAVE A TRAILING SLASH. Example: "https://friendly-doodle-x5x6qvv74vr6h655x-8000.app.github.dev"
    const FASTAPI_BASE_URL = "https://super-fishstick-7vwq4pp6xrjvhppgw-8000.app.github.dev"; // <--- REMOVE TRAILING SLASH IF PRESENT

    const handleRegister = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError('');
        setMessage('');
        try {
            // Ensure auth is available
            if (!auth) {
                throw new Error("Firebase Auth is not initialized.");
            }

            const response = await fetch(`${FASTAPI_BASE_URL}/register`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, password, username }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Registration failed.');
            }

            const data = await response.json();
            setMessage(data.message || 'Registration successful! Please log in.');
            await logEvent('user_registration', { email: email, username: username, uid: data.uid }, true);
            onRegisterSuccess(); // Notify parent App component
        } catch (err) {
            console.error("Registration error:", err);
            setError(err.message || 'An unexpected error occurred during registration.');
            await logEvent('user_registration', { email: email, username: username, error: err.message }, false, err.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="flex items-center justify-center min-h-screen bg-gradient-to-br from-indigo-50 to-purple-100 p-4">
            <div className="bg-white p-8 rounded-xl shadow-2xl w-full max-w-md transform transition-all duration-300 hover:scale-[1.01]">
                <h2 className="text-4xl font-extrabold text-indigo-800 mb-6 text-center">Register</h2>
                {error && <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative mb-4 text-center">{error}</div>}
                {message && <div className="bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded relative mb-4 text-center">{message}</div>}
                <form onSubmit={handleRegister} className="space-y-6">
                    <div>
                        <label className="block text-gray-700 text-sm font-bold mb-2" htmlFor="username">Username</label>
                        <input
                            type="text"
                            id="username"
                            value={username}
                            onChange={(e) => setUsername(e.target.value)}
                            className="shadow appearance-none border rounded-md w-full py-3 px-4 text-gray-700 leading-tight focus:outline-none focus:ring-2 focus:ring-indigo-500 transition duration-200"
                            placeholder="Your Username"
                            required
                        />
                    </div>
                    <div>
                        <label className="block text-gray-700 text-sm font-bold mb-2" htmlFor="email">Email</label>
                        <input
                            type="email"
                            id="email"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            className="shadow appearance-none border rounded-md w-full py-3 px-4 text-gray-700 leading-tight focus:outline-none focus:ring-2 focus:ring-indigo-500 transition duration-200"
                            placeholder="your@example.com"
                            required
                        />
                    </div>
                    <div>
                        <label className="block text-gray-700 text-sm font-bold mb-2" htmlFor="password">Password</label>
                        <input
                            type="password"
                            id="password"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            className="shadow appearance-none border rounded-md w-full py-3 px-4 text-gray-700 mb-3 leading-tight focus:outline-none focus:ring-2 focus:ring-indigo-500 transition duration-200"
                            placeholder="********"
                            required
                            minLength="6"
                        />
                    </div>
                    <button
                        type="submit"
                        className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-3 px-4 rounded-md shadow-lg transform hover:scale-105 transition duration-300 flex items-center justify-center"
                        disabled={loading}
                    >
                        {loading ? (
                            <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                            </svg>
                        ) : 'Register'}
                    </button>
                </form>
                <div className="mt-6 text-center">
                    <p className="text-gray-600">Already have an account?</p>
                    <button
                        onClick={onNavigateToLogin}
                        className="text-indigo-600 hover:text-indigo-800 font-semibold mt-2 transition duration-300"
                    >
                        Login here
                    </button>
                </div>
            </div>
        </div>
    );
};

// --- UserProfile Component ---
// Now receives 'auth' prop
const UserProfile = ({ userId, auth }) => {
    // IMPORTANT: Replace this with your actual Codespace URL or deployed backend URL.
    // ENSURE IT DOES NOT HAVE A TRAILING SLASH. Example: "https://friendly-doodle-x5x6qvv74vr6h655x-8000.app.github.dev"
    const FASTAPI_BASE_URL = "https://super-fishstick-7vwq4pp6xrjvhppgw-8000.app.github.dev"; // <--- REMOVE TRAILING SLASH IF PRESENT

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
            setError("User not authenticated. Please log in.");
            setLoading(false);
            return;
        }
        // Ensure auth is available before trying to get ID token
        if (!auth || !auth.currentUser) {
            setError("Authentication not ready. Please wait or log in again.");
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
                    'Authorization': `Bearer ${idToken}`
                }
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to fetch user profile.');
            }

            const data = await response.json();
            setUserData(data.user || data); // Backend returns {"user": profile_data}
            await logEvent('user_profile_view', { user_id: userId, status: 'success' });
        } catch (err) {
            console.error("Error fetching user profile:", err);
            setError(err.message);
            await logEvent('user_profile_view', { user_id: userId, status: 'failure', error: err.message });
        } finally {
            setLoading(false);
        }
    }, [userId, auth, FASTAPI_BASE_URL]);

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
            const idToken = await auth.currentUser?.getIdToken(true);
            if (!idToken) {
                throw new Error("No authentication token available. Please log in.");
            }

            const response = await fetch(`${FASTAPI_BASE_URL}/user/profile`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${idToken}`
                },
                body: JSON.stringify({
                    phone: userData.phone,
                    address: userData.address,
                    bio: userData.bio,
                    username: userData.username // Include username in update as it's editable
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
            fetchUserProfile(); // Re-fetch to get the latest data
        } catch (err) {
            console.error("Error updating user profile:", err);
            setError(err.message);
            await logEvent('user_profile_update', { user_id: userId, status: 'failure', error: err.message });
        } finally {
            setLoading(false);
        }
    };

    const handleCancel = () => {
        setIsEditing(false);
        setMessage('Edit cancelled.');
        fetchUserProfile(); // Discard changes by re-fetching
    };

    return (
        <div className="max-w-4xl mx-auto bg-white p-8 rounded-lg shadow-xl my-10 animate-fade-in">
            <h2 className="text-4xl font-extrabold text-indigo-800 mb-8 text-center">User Profile</h2>

            {message && (
                <div className={`p-4 mb-6 rounded-md text-center ${error ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'}`}>
                    {message}
                </div>
            )}

            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
                <div className="space-y-6">
                    <div>
                        <label className="block text-gray-700 text-sm font-bold mb-2" htmlFor="username">
                            Username
                        </label>
                        <input
                            type="text"
                            id="username"
                            name="username"
                            value={userData.username}
                            onChange={handleChange}
                            readOnly={!isEditing}
                            className={`shadow appearance-none border rounded-md w-full py-3 px-4 text-gray-700 leading-tight focus:outline-none focus:ring-2 focus:ring-indigo-500 transition duration-200 ${isEditing ? 'bg-white' : 'bg-gray-100 cursor-not-allowed'}`}
                        />
                    </div>
                    <div>
                        <label className="block text-gray-700 text-sm font-bold mb-2" htmlFor="email">
                            Email
                        </label>
                        <input
                            type="email"
                            id="email"
                            name="email"
                            value={userData.email}
                            readOnly // Email is typically managed by Firebase Auth directly, not via profile update
                            className={`shadow appearance-none border rounded-md w-full py-3 px-4 text-gray-700 leading-tight focus:outline-none focus:ring-2 focus:ring-indigo-500 transition duration-200 bg-gray-100 cursor-not-allowed`}
                        />
                    </div>
                    <div>
                        <label className="block text-gray-700 text-sm font-bold mb-2" htmlFor="phone">
                            Phone
                        </label>
                        <input
                            type="tel"
                            id="phone"
                            name="phone"
                            value={userData.phone}
                            onChange={handleChange}
                            readOnly={!isEditing}
                            className={`shadow appearance-none border rounded-md w-full py-3 px-4 text-gray-700 leading-tight focus:outline-none focus:ring-2 focus:ring-indigo-500 transition duration-200 ${isEditing ? 'bg-white' : 'bg-gray-100 cursor-not-allowed'}`}
                        />
                    </div>
                </div>
                <div className="space-y-6">
                    <div>
                        <label className="block text-gray-700 text-sm font-bold mb-2" htmlFor="address">
                            Address
                        </label>
                        <input
                            type="text"
                            id="address"
                            name="address"
                            value={userData.address}
                            onChange={handleChange}
                            readOnly={!isEditing}
                            className={`shadow appearance-none border rounded-md w-full py-3 px-4 text-gray-700 leading-tight focus:outline-none focus:ring-2 focus:ring-indigo-500 transition duration-200 ${isEditing ? 'bg-white' : 'bg-gray-100 cursor-not-allowed'}`}
                        />
                    </div>
                    <div>
                        <label className="block text-gray-700 text-sm font-bold mb-2" htmlFor="bio">
                            Bio
                        </label>
                        <textarea
                            id="bio"
                            name="bio"
                            value={userData.bio}
                            onChange={handleChange}
                            readOnly={!isEditing}
                            rows="4"
                            className={`shadow appearance-none border rounded-md w-full py-3 px-4 text-gray-700 leading-tight focus:outline-none focus:ring-2 focus:ring-indigo-500 transition duration-200 ${isEditing ? 'bg-white' : 'bg-gray-100 cursor-not-allowed'}`}
                        ></textarea>
                    </div>
                    <div>
                        <label className="block text-gray-700 text-sm font-bold mb-2" htmlFor="tier">
                            Tier
                        </label>
                        <input
                            type="text"
                            id="tier"
                            name="tier"
                            value={userData.tier}
                            readOnly // Tier is read-only, managed by admin
                            className="shadow appearance-none border rounded-md w-full py-3 px-4 text-gray-700 leading-tight focus:outline-none focus:ring-2 focus:ring-gray-300 bg-gray-100 cursor-not-allowed"
                        />
                    </div>
                </div>
            </div>

            <div className="flex justify-end space-x-4 mt-8">
                {loading ? (
                    <button
                        className="bg-indigo-400 text-white font-bold py-3 px-6 rounded-md shadow-lg flex items-center justify-center"
                        disabled
                    >
                        <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        Processing...
                    </button>
                ) : !isEditing ? (
                    <button
                        onClick={() => setIsEditing(true)}
                        className="bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-3 px-6 rounded-md shadow-lg transition duration-300 flex items-center"
                    >
                        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="lucide lucide-user-pen mr-2"><path d="M12 20v-2c0-1.1.9-2 2-2h4a2 2 0 0 1 2 2v2"/><path d="M18.42 1.41a2 2 0 0 1 2.83 0l.59.59a2 2 0 0 1 0 2.83L18 8l-5-5 5.42-5.42Z"/><path d="M12 4V2"/><path d="M10 20H6a2 2 0 0 0-2 2v2"/><circle cx="8" cy="7" r="4"/></svg>
                        Edit Profile
                    </button>
                ) : (
                    <>
                        <button
                            onClick={handleSave}
                            className="bg-green-600 hover:bg-green-700 text-white font-bold py-3 px-6 rounded-md shadow-lg transition duration-300 flex items-center"
                        >
                            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="lucide lucide-save mr-2"><path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2Z"/><polyline points="17 21 17 13 7 13 7 21"/><path d="M7 3v4a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2V3"/></svg>
                            Save Changes
                        </button>
                        <button
                            onClick={handleCancel}
                            className="bg-gray-400 hover:bg-gray-500 text-white font-bold py-3 px-6 rounded-md shadow-lg transition duration-300 flex items-center"
                        >
                            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="lucide lucide-x-circle mr-2"><circle cx="12" cy="12" r="10"/><path d="m15 9-6 6"/><path d="m9 9 6 6"/></svg>
                            Cancel
                        </button>
                    </>
                )}
            </div>
        </div>
    );
};

// --- AnalyticsDashboard Component ---
// Now receives 'db' and 'auth' props
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
            const analyticsCollectionRef = collection(db, `artifacts/${appId}/public/data/analytics_events`);
            let q = query(analyticsCollectionRef);

            if (filterEventType) {
                q = query(q, where('event_type', '==', filterEventType));
            }
            if (filterUserId) {
                q = query(q, where('user_id', '==', filterUserId));
            }

            const querySnapshot = await getDocs(q);
            let fetchedEvents = querySnapshot.docs.map(doc => ({ id: doc.id, ...doc.data() }));

            // In-memory date filtering and sorting
            if (filterStartDate) {
                const start = new Date(filterStartDate).toISOString();
                fetchedEvents = fetchedEvents.filter(event => event.timestamp >= start);
            }
            if (filterEndDate) {
                const end = new Date(filterEndDate).toISOString();
                fetchedEvents = fetchedEvents.filter(event => event.timestamp <= end);
            }

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

    // Prepare data for charts (simplified, no recharts import here)
    const getChartData = () => {
        const dailyCounts = {};
        const toolSuccessCounts = {};
        const toolFailureCounts = {};

        events.forEach(event => {
            if (event.timestamp) {
                const dateKey = new Date(event.timestamp).toISOString().split('T')[0]; //YYYY-MM-DD
                dailyCounts[dateKey] = (dailyCounts[dateKey] || 0) + 1;
            }

            if (event.event_type === 'tool_usage' && event.details && event.details.tool_name) {
                const toolName = event.details.tool_name;
                if (event.success) {
                    toolSuccessCounts[toolName] = (toolSuccessCounts[toolName] || 0) + 1;
                } else {
                    toolFailureCounts[toolName] = (toolFailureCounts[toolName] || 0) + 1;
                }
            }
        });

        const dailyChartData = Object.keys(dailyCounts).sort().map(date => ({
            date,
            count: dailyCounts[date]
        }));

        const toolUsageChartData = Object.keys(toolSuccessCounts).map(toolName => ({
            toolName,
            success: toolSuccessCounts[toolName],
            failure: toolFailureCounts[toolName] || 0
        })).sort((a, b) => (b.success + b.failure) - (a.success + a.failure));

        return { dailyChartData, toolUsageChartData };
    };

    const { dailyChartData, toolUsageChartData } = getChartData();

    const handleExport = () => {
        const headers = ["Event ID", "Timestamp", "User ID", "Event Type", "Tool Name", "Success", "Error Message", "Details (JSON)"];
        const csvRows = [headers.join(',')];

        events.forEach(event => {
            const row = [
                event.id,
                event.timestamp ? new Date(event.timestamp).toISOString() : '',
                event.user_id || 'N/A',
                event.event_type || 'N/A',
                event.details?.tool_name || 'N/A',
                event.success ? 'True' : 'False',
                event.error_message || '',
                JSON.stringify(event.details || {})
            ];
            csvRows.push(row.map(item => `"${String(item).replace(/"/g, '""')}"`).join(','));
        });

        const csvString = csvRows.join('\n');
        const blob = new Blob([csvString], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.setAttribute('download', `analytics_data_${new Date().toISOString().split('T')[0]}.csv`);
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };

    const uniqueEventTypes = [...new Set(events.map(e => e.event_type))];
    const uniqueUserIds = [...new Set(events.map(e => e.user_id))];

    return (
        <div className="max-w-6xl mx-auto bg-white p-8 rounded-lg shadow-xl mt-10">
            <h2 className="text-4xl font-extrabold text-indigo-800 mb-8 text-center">Analytics Dashboard</h2>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
                <select
                    value={filterEventType}
                    onChange={handleFilterChange(setFilterEventType)}
                    className="p-3 border border-gray-300 rounded-md focus:outline-none focus:border-indigo-400"
                >
                    <option value="">All Event Types</option>
                    {uniqueEventTypes.map(type => <option key={type} value={type}>{type}</option>)}
                </select>
                <select
                    value={filterUserId}
                    onChange={handleFilterChange(setFilterUserId)}
                    className="p-3 border border-gray-300 rounded-md focus:outline-none focus:border-indigo-400"
                >
                    <option value="">All User IDs</option>
                    {uniqueUserIds.map(id => <option key={id} value={id}>{id}</option>)}
                </select>
                <input
                    type="date"
                    value={filterStartDate}
                    onChange={handleFilterChange(setFilterStartDate)}
                    className="p-3 border border-gray-300 rounded-md focus:outline-none focus:border-indigo-400"
                />
                <input
                    type="date"
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

            {loading ? (
                <div className="text-center py-20 text-xl text-indigo-600">Loading analytics data...</div>
            ) : error ? (
                <div className="text-center py-20 text-red-500 text-xl">Error: {error}</div>
            ) : events.length === 0 ? (
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
const App = () => {
    const [currentPage, setCurrentPage] = useState('login'); // Start at login page
    const [currentUserId, setCurrentUserId] = useState(null);
    const [isAuthReady, setIsAuthReady] = useState(false);

    // Use useState to hold Firebase instances
    const [firebaseApp, setFirebaseApp] = useState(null);
    const [firestoreDb, setFirestoreDb] = useState(null);
    const [firebaseAuth, setFirebaseAuth] = useState(null);

    // Firebase client-side configuration
    const appId = typeof __app_id !== 'undefined' ? __app_id : 'default-app-id';
    const firebaseConfig = typeof __firebase_config !== 'undefined' ? JSON.parse(__firebase_config) : {};

    // Initialize Firebase once on component mount
    useEffect(() => {
        if (Object.keys(firebaseConfig).length === 0) {
            console.warn("Firebase config not found. Application may not function correctly.");
            setIsAuthReady(true); // Allow app to proceed even if Firebase isn't configured
            return;
        }

        try {
            const app = initializeApp(firebaseConfig);
            const db = getFirestore(app);
            const auth = getAuth(app);

            setFirebaseApp(app);
            setFirestoreDb(db);
            setFirebaseAuth(auth);
            console.log("Firebase initialized successfully in App.jsx (via useEffect)");
        } catch (error) {
            console.error("Error initializing Firebase in App.jsx:", error);
            setIsAuthReady(true); // Mark as ready even on error to prevent infinite loading
        }
    }, [appId, JSON.stringify(firebaseConfig)]); // Re-run if config changes (unlikely in this context)

    // Auth state listener (depends on firebaseAuth being set)
    useEffect(() => {
        if (!firebaseAuth || !firestoreDb) {
            console.warn("Firebase Auth or Firestore not available for auth listener. Waiting for initialization.");
            return;
        }

        const unsubscribe = onAuthStateChanged(firebaseAuth, async (user) => {
            if (user) {
                setCurrentUserId(user.uid);
                // If coming from login/register, navigate to home/profile
                if (currentPage === 'login' || currentPage === 'register') {
                    setCurrentPage('home');
                }
            } else {
                setCurrentUserId(null);
                // If not authenticated, always show login page unless on register page
                if (currentPage !== 'register') {
                    setCurrentPage('login');
                }
            }
            setIsAuthReady(true);
            // Initialize analytics once auth is ready and we have a potential user ID
            initializeAnalytics(firestoreDb, firebaseAuth, appId, user?.uid || 'anonymous');
        });

        return () => unsubscribe();
    }, [firebaseAuth, firestoreDb, appId, currentPage]); // Depend on firebaseAuth and firestoreDb

    // Log page views when currentPage changes (after analytics is initialized)
    useEffect(() => {
        if (isAuthReady && currentUserId) {
            (async () => {
                await logEvent('page_view', { page_name: currentPage, user_id: currentUserId });
                console.log(`Analytics: Logged page_view for '${currentPage}' by user '${currentUserId}'`);
            })();
        }
    }, [currentPage, isAuthReady, currentUserId]);


    const handleLogout = async () => {
        if (firebaseAuth) {
            try {
                await firebaseAuth.signOut();
                console.log("User logged out.");
                await logEvent('user_logout', { user_id: currentUserId }, true);
                setCurrentUserId(null);
                setCurrentPage('login'); // Redirect to login page after logout
            } catch (error) {
                console.error("Error logging out:", error);
                await logEvent('user_logout', { user_id: currentUserId, error: error.message }, false, error.message);
            }
        }
    };

    const renderPage = useCallback(() => {
        if (!isAuthReady || !firebaseApp || !firestoreDb || !firebaseAuth) {
            return (
                <div className="flex items-center justify-center min-h-screen bg-gradient-to-br from-indigo-50 to-purple-100">
                    <div className="text-center py-20 text-xl text-indigo-600 flex items-center">
                        <svg className="animate-spin -ml-1 mr-3 h-8 w-8 text-indigo-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        Initializing application...
                    </div>
                </div>
            );
        }

        // Pass auth and db instances to child components
        switch (currentPage) {
            case 'login':
                return <LoginPage onLoginSuccess={() => setCurrentPage('home')} onNavigateToRegister={() => setCurrentPage('register')} auth={firebaseAuth} />;
            case 'register':
                return <RegisterPage onRegisterSuccess={() => setCurrentPage('login')} onNavigateToLogin={() => setCurrentPage('login')} auth={firebaseAuth} />;
            case 'home':
                return (
                    <div className="text-center py-20 bg-gradient-to-br from-indigo-50 to-purple-100 min-h-[calc(100vh-80px)] flex flex-col justify-center items-center">
                        <h1 className="text-5xl font-extrabold text-indigo-800 mb-6 animate-fade-in-down">
                            Welcome to Intelli-Agent!
                        </h1>
                        <p className="text-xl text-gray-700 mb-8 animate-fade-in-up">
                            Your smart assistant for everything. Explore your profile or view analytics.
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
                return <AnalyticsDashboard db={firestoreDb} auth={firebaseAuth} appId={appId} currentUserId={currentUserId} />;
            case 'profile':
                return <UserProfile userId={currentUserId} auth={firebaseAuth} />;
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
    }, [currentPage, isAuthReady, currentUserId, firebaseApp, firestoreDb, firebaseAuth, appId]);

    return (
        <div className="min-h-screen bg-gradient-to-br from-indigo-50 to-purple-100 font-inter">
            <header className="bg-indigo-800 text-white p-4 shadow-md">
                <div className="container mx-auto flex justify-between items-center">
                    <h1 className="text-3xl font-bold">Intelli-Agent</h1>
                    <nav>
                        <ul className="flex justify-center space-x-6">
                            {currentUserId && ( // Only show navigation if logged in
                                <>
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
                                    <li>
                                        <button
                                            onClick={handleLogout}
                                            className="bg-red-500 hover:bg-red-600 text-white font-bold py-2 px-4 rounded-full shadow-md transition duration-300"
                                        >
                                            Logout
                                        </button>
                                    </li>
                                </>
                            )}
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

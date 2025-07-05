import React, { useState, useEffect, useCallback } from 'react';
import { initializeApp } from 'firebase/app';
import { getAuth, signInAnonymously, signInWithCustomToken, onAuthStateChanged } from 'firebase/auth';
import { getFirestore, collection, query, orderBy, where, getDocs } from 'firebase/firestore';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { CalendarDays, Filter, RefreshCcw, Download } from 'lucide-react';

// Ensure __app_id and __firebase_config are available in the environment
const appId = typeof __app_id !== 'undefined' ? __app_id : 'default-app-id';
const firebaseConfig = typeof __firebase_config !== 'undefined' ? JSON.parse(__firebase_config) : {};

// Initialize Firebase outside the component to prevent re-initialization
let app, db, auth;
if (Object.keys(firebaseConfig).length > 0) {
    app = initializeApp(firebaseConfig);
    db = getFirestore(app);
    auth = getAuth(app);
} else {
    console.warn("Firebase config not found. Analytics Dashboard will not function.");
}

const AnalyticsDashboard = () => {
    const [events, setEvents] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [userId, setUserId] = useState(null);
    const [isAuthReady, setIsAuthReady] = useState(false);
    const [startDate, setStartDate] = useState('');
    const [endDate, setEndDate] = useState('');
    const [filterEventType, setFilterEventType] = useState('');
    const [filterToolName, setFilterToolName] = useState('');
    const [filterSuccess, setFilterSuccess] = useState('all'); // 'all', 'true', 'false'

    // Authentication Listener
    useEffect(() => {
        if (!auth) {
            setIsAuthReady(true); // No auth configured, proceed without auth
            return;
        }

        const unsubscribe = onAuthStateChanged(auth, async (user) => {
            if (user) {
                setUserId(user.uid);
            } else {
                // Sign in anonymously if no user is authenticated
                try {
                    if (typeof __initial_auth_token !== 'undefined' && __initial_auth_token) {
                        await signInWithCustomToken(auth, __initial_auth_token);
                    } else {
                        await signInAnonymously(auth);
                    }
                    setUserId(auth.currentUser?.uid || crypto.randomUUID()); // Fallback if anonymous uid is null
                } catch (anonError) {
                    console.error("Error signing in anonymously:", anonError);
                    setError("Failed to authenticate for analytics. Please try again.");
                }
            }
            setIsAuthReady(true);
        });

        return () => unsubscribe();
    }, []);

    const fetchAnalyticsData = useCallback(async () => {
        if (!db || !isAuthReady) {
            console.log("Firestore not initialized or auth not ready.");
            return;
        }

        setLoading(true);
        setError(null);
        try {
            // Firestore security rules:
            // Public data: /artifacts/{appId}/public/data/{your_collection_name}
            // Private data: /artifacts/{appId}/users/{userId}/{your_collection_name}
            // For analytics, we'll store in a public-like collection under the app ID
            // but ensure only authenticated users can read.
            const analyticsCollectionRef = collection(db, `artifacts/${appId}/public/data/analytics_events`);
            let q = query(analyticsCollectionRef, orderBy('timestamp', 'desc'));

            // Apply filters
            if (startDate) {
                const startTimestamp = new Date(startDate).getTime();
                q = query(q, where('timestamp', '>=', startTimestamp));
            }
            if (endDate) {
                const endTimestamp = new Date(endDate).setHours(23, 59, 59, 999); // End of day
                q = query(q, where('timestamp', '<=', endTimestamp));
            }
            if (filterEventType && filterEventType !== 'all') {
                q = query(q, where('event_type', '==', filterEventType));
            }
            if (filterToolName) {
                q = query(q, where('details.tool_name', '==', filterToolName));
            }
            if (filterSuccess !== 'all') {
                q = query(q, where('success', '==', filterSuccess === 'true'));
            }

            const querySnapshot = await getDocs(q);
            const fetchedEvents = querySnapshot.docs.map(doc => ({
                id: doc.id,
                ...doc.data(),
                timestamp: doc.data().timestamp ? new Date(doc.data().timestamp) : null // Convert timestamp back to Date object
            }));
            setEvents(fetchedEvents);
        } catch (err) {
            console.error("Error fetching analytics data:", err);
            setError("Failed to load analytics data. " + err.message);
        } finally {
            setLoading(false);
        }
    }, [db, isAuthReady, startDate, endDate, filterEventType, filterToolName, filterSuccess]);

    // Fetch data when auth is ready or filters change
    useEffect(() => {
        if (isAuthReady) {
            fetchAnalyticsData();
        }
    }, [isAuthReady, fetchAnalyticsData]);

    // Prepare data for charts
    const getChartData = () => {
        const dailyCounts = {};
        const toolSuccessCounts = {};
        const toolFailureCounts = {};

        events.forEach(event => {
            if (event.timestamp) {
                const dateKey = event.timestamp.toISOString().split('T')[0]; // YYYY-MM-DD
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
        })).sort((a, b) => (b.success + b.failure) - (a.success + a.failure)); // Sort by total usage

        return { dailyChartData, toolUsageChartData };
    };

    const { dailyChartData, toolUsageChartData } = getChartData();

    const handleExport = () => {
        const headers = ["Event ID", "Timestamp", "User ID", "Event Type", "Tool Name", "Success", "Error Message", "Tool Params (JSON)"];
        const csvRows = [headers.join(',')];

        events.forEach(event => {
            const row = [
                event.id,
                event.timestamp ? event.timestamp.toISOString() : '',
                event.user_id || 'N/A',
                event.event_type || 'N/A',
                event.details?.tool_name || 'N/A',
                event.success ? 'True' : 'False',
                event.error_message || '',
                JSON.stringify(event.details?.tool_params || {})
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

    // Get unique tool names for filtering
    const uniqueToolNames = [...new Set(events.filter(e => e.event_type === 'tool_usage' && e.details?.tool_name).map(e => e.details.tool_name))];

    return (
        <div className="p-6 bg-gray-100 min-h-screen font-inter text-gray-800 rounded-lg shadow-inner">
            <h1 className="text-4xl font-extrabold text-center text-indigo-800 mb-8 flex items-center justify-center">
                <CalendarDays className="mr-3 text-indigo-600" size={36} />
                Analytics Dashboard
            </h1>

            <div className="bg-white p-6 rounded-xl shadow-lg mb-8">
                <h2 className="text-2xl font-bold text-indigo-700 mb-4 flex items-center">
                    <Filter className="mr-2 text-indigo-500" size={24} />
                    Filters
                </h2>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
                    <div>
                        <label htmlFor="startDate" className="block text-sm font-medium text-gray-700 mb-1">Start Date</label>
                        <input
                            type="date"
                            id="startDate"
                            value={startDate}
                            onChange={(e) => setStartDate(e.target.value)}
                            className="w-full p-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500 shadow-sm"
                        />
                    </div>
                    <div>
                        <label htmlFor="endDate" className="block text-sm font-medium text-gray-700 mb-1">End Date</label>
                        <input
                            type="date"
                            id="endDate"
                            value={endDate}
                            onChange={(e) => setEndDate(e.target.value)}
                            className="w-full p-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500 shadow-sm"
                        />
                    </div>
                    <div>
                        <label htmlFor="eventType" className="block text-sm font-medium text-gray-700 mb-1">Event Type</label>
                        <select
                            id="eventType"
                            value={filterEventType}
                            onChange={(e) => setFilterEventType(e.target.value)}
                            className="w-full p-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500 shadow-sm"
                        >
                            <option value="all">All</option>
                            <option value="tool_usage">Tool Usage</option>
                            <option value="page_view">Page View</option>
                            <option value="ui_interaction">UI Interaction</option>
                        </select>
                    </div>
                    <div>
                        <label htmlFor="toolName" className="block text-sm font-medium text-gray-700 mb-1">Tool Name</label>
                        <select
                            id="toolName"
                            value={filterToolName}
                            onChange={(e) => setFilterToolName(e.target.value)}
                            className="w-full p-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500 shadow-sm"
                        >
                            <option value="">All Tools</option>
                            {uniqueToolNames.map(tool => (
                                <option key={tool} value={tool}>{tool}</option>
                            ))}
                        </select>
                    </div>
                    <div>
                        <label htmlFor="successFilter" className="block text-sm font-medium text-gray-700 mb-1">Success Status</label>
                        <select
                            id="successFilter"
                            value={filterSuccess}
                            onChange={(e) => setFilterSuccess(e.target.value)}
                            className="w-full p-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500 shadow-sm"
                        >
                            <option value="all">All</option>
                            <option value="true">Success</option>
                            <option value="false">Failure</option>
                        </select>
                    </div>
                </div>
                <button
                    onClick={fetchAnalyticsData}
                    className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-2 px-4 rounded-md shadow-md transition duration-300 flex items-center justify-center"
                    disabled={loading}
                >
                    <RefreshCcw className="mr-2" size={20} />
                    {loading ? 'Loading...' : 'Apply Filters / Refresh Data'}
                </button>
            </div>

            {error && (
                <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded-xl relative mb-8 shadow-md" role="alert">
                    <strong className="font-bold">Error!</strong>
                    <span className="block sm:inline"> {error}</span>
                </div>
            )}

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
                <div className="bg-white p-6 rounded-xl shadow-lg">
                    <h2 className="text-2xl font-bold text-indigo-700 mb-4">Events Over Time</h2>
                    <ResponsiveContainer width="100%" height={300}>
                        <LineChart data={dailyChartData} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                            <XAxis dataKey="date" />
                            <YAxis />
                            <Tooltip />
                            <Legend />
                            <Line type="monotone" dataKey="count" stroke="#6366f1" activeDot={{ r: 8 }} name="Total Events" />
                        </LineChart>
                    </ResponsiveContainer>
                </div>

                <div className="bg-white p-6 rounded-xl shadow-lg">
                    <h2 className="text-2xl font-bold text-indigo-700 mb-4">Tool Usage Success/Failure</h2>
                    <ResponsiveContainer width="100%" height={300}>
                        <BarChart data={toolUsageChartData} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                            <XAxis dataKey="toolName" angle={-45} textAnchor="end" height={80} interval={0} />
                            <YAxis />
                            <Tooltip />
                            <Legend />
                            <Bar dataKey="success" stackId="a" fill="#4CAF50" name="Success" radius={[10, 10, 0, 0]} />
                            <Bar dataKey="failure" stackId="a" fill="#F44336" name="Failure" radius={[0, 0, 10, 10]} />
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            </div>

            <div className="bg-white p-6 rounded-xl shadow-lg mb-8">
                <h2 className="text-2xl font-bold text-indigo-700 mb-4 flex items-center justify-between">
                    All Analytics Events
                    <button
                        onClick={handleExport}
                        className="bg-green-500 hover:bg-green-600 text-white font-bold py-2 px-4 rounded-md shadow-md transition duration-300 flex items-center"
                    >
                        <Download className="mr-2" size={20} />
                        Export to CSV
                    </button>
                </h2>
                {loading ? (
                    <div className="text-center py-10 text-lg text-gray-600">Loading events...</div>
                ) : events.length === 0 ? (
                    <div className="text-center py-10 text-lg text-gray-600">No events found for the selected filters.</div>
                ) : (
                    <div className="overflow-x-auto rounded-lg border border-gray-200 shadow-sm">
                        <table className="min-w-full divide-y divide-gray-200">
                            <thead className="bg-gray-50">
                                <tr>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Timestamp</th>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">User ID</th>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Event Type</th>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Tool Name</th>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Success</th>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Message</th>
                                </tr>
                            </thead>
                            <tbody className="bg-white divide-y divide-gray-200">
                                {events.map((event) => (
                                    <tr key={event.id} className="hover:bg-gray-50">
                                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                                            {event.timestamp ? event.timestamp.toLocaleString() : 'N/A'}
                                        </td>
                                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                            {event.user_id || 'Anonymous'}
                                        </td>
                                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                                            <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full 
                                                ${event.event_type === 'tool_usage' ? 'bg-blue-100 text-blue-800' :
                                                  event.event_type === 'page_view' ? 'bg-purple-100 text-purple-800' :
                                                  event.event_type === 'ui_interaction' ? 'bg-yellow-100 text-yellow-800' : 'bg-gray-100 text-gray-800'}`}>
                                                {event.event_type}
                                            </span>
                                        </td>
                                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                                            {event.details?.tool_name || 'N/A'}
                                        </td>
                                        <td className="px-6 py-4 whitespace-nowrap text-sm">
                                            <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full 
                                                ${event.success ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
                                                {event.success ? 'Yes' : 'No'}
                                            </span>
                                        </td>
                                        <td className="px-6 py-4 text-sm text-gray-500 max-w-xs truncate">
                                            {event.error_message || event.details?.message || 'N/A'}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}
            </div>
        </div>
    );
};

export default AnalyticsDashboard;

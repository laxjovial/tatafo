import React, { useState, useEffect } from 'react';
import { User, Save, XCircle, Loader2 } from 'lucide-react'; // Import Loader2 for loading indicator
import { logEvent } from './utils/analytics_tracker'; // Ensure correct path to logEvent

// IMPORTANT: Replace this with your actual Codespace URL or deployed backend URL
const FASTAPI_BASE_URL = "https://friendly-doodle-x5x6qvv74vr6h655x-8000.app.github.dev/"; // Use your actual Codespace URL here!

const UserProfile = ({ userId }) => {
    const [profile, setProfile] = useState({
        username: '',
        email: '',
        phone: '',
        address: '',
        bio: '',
        tier: ''
    });
    const [isEditing, setIsEditing] = useState(false);
    const [statusMessage, setStatusMessage] = useState('');
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    // Function to fetch user profile
    const fetchUserProfile = async (id) => {
        if (!id) {
            console.warn("User ID is null or undefined, cannot fetch profile.");
            setLoading(false);
            return;
        }
        setLoading(true);
        setError(null);
        try {
            const response = await fetch(`${FASTAPI_BASE_URL}/profile/${id}`);
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to fetch user profile');
            }
            const data = await response.json();
            setProfile(data);
            console.log(`User profile fetched for ${id}:`, data);
            await logEvent('data_fetch', { component: 'UserProfile', action: 'Fetch Profile', user_id: id, success: true });
        } catch (err) {
            console.error('Error fetching user profile:', err);
            setError(err.message);
            await logEvent('data_fetch', { component: 'UserProfile', action: 'Fetch Profile', user_id: id, success: false, error_message: err.message });
        } finally {
            setLoading(false);
        }
    };

    // Log page view and fetch profile when component mounts or userId changes
    useEffect(() => {
        if (userId) {
            (async () => {
                await logEvent('page_view', { page_name: 'UserProfile', user_id: userId });
                console.log(`Analytics: Logged page_view for 'UserProfile' by user '${userId}'`);
                fetchUserProfile(userId);
            })();
        }
    }, [userId]); // Depend on userId to refetch if it changes

    // Effect for status message display
    useEffect(() => {
        if (statusMessage) {
            const timer = setTimeout(() => {
                setStatusMessage('');
            }, 3000); // Clear message after 3 seconds
            return () => clearTimeout(timer);
        }
    }, [statusMessage]);

    const handleChange = (e) => {
        const { name, value } = e.target;
        setProfile(prevProfile => ({
            ...prevProfile,
            [name]: value
        }));
    };

    const handleEditClick = () => {
        setIsEditing(true);
        setStatusMessage(''); // Clear any previous status messages
        logEvent('ui_interaction', { component: 'UserProfile', action: 'Edit Profile Click', user_id: userId, success: true });
    };

    const handleSave = async () => {
        setLoading(true);
        setError(null);
        try {
            const response = await fetch(`${FASTAPI_BASE_URL}/profile/update/${userId}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(profile),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to save profile');
            }

            setIsEditing(false);
            setStatusMessage('Profile saved successfully!');
            console.log('Profile saved:', profile);
            await logEvent('ui_interaction', {
                component: 'UserProfile',
                action: 'Save Profile',
                details: { userId, changes: profile },
                user_id: userId,
                success: true
            });
        } catch (err) {
            console.error('Error saving profile:', err);
            setError(err.message);
            setStatusMessage(`Error: ${err.message}`);
            await logEvent('ui_interaction', {
                component: 'UserProfile',
                action: 'Save Profile',
                details: { userId, changes: profile },
                user_id: userId,
                success: false,
                error_message: err.message
            });
        } finally {
            setLoading(false);
        }
    };

    const handleCancel = () => {
        setIsEditing(false);
        setStatusMessage('Edit cancelled.');
        // Re-fetch original profile data to discard changes
        fetchUserProfile(userId);
        logEvent('ui_interaction', { component: 'UserProfile', action: 'Cancel Edit', user_id: userId, success: true });
    };

    if (loading) {
        return (
            <div className="flex justify-center items-center h-64">
                <Loader2 className="h-12 w-12 animate-spin text-indigo-600" />
                <p className="ml-4 text-xl text-indigo-600">Loading profile...</p>
            </div>
        );
    }

    if (error) {
        return (
            <div className="text-center py-20">
                <p className="text-red-500 text-xl">Error: {error}</p>
                <button
                    onClick={() => fetchUserProfile(userId)}
                    className="mt-4 bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-2 px-6 rounded-md shadow-lg transition duration-300"
                >
                    Retry
                </button>
            </div>
        );
    }

    return (
        <div className="max-w-4xl mx-auto bg-white p-8 rounded-lg shadow-xl my-10 animate-fade-in">
            <h2 className="text-4xl font-extrabold text-indigo-800 mb-8 text-center">User Profile</h2>

            {statusMessage && (
                <div className={`p-4 mb-6 rounded-md text-center ${error ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'}`}>
                    {statusMessage}
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
                            value={profile.username}
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
                            value={profile.email}
                            onChange={handleChange}
                            readOnly={!isEditing}
                            className={`shadow appearance-none border rounded-md w-full py-3 px-4 text-gray-700 leading-tight focus:outline-none focus:ring-2 focus:ring-indigo-500 transition duration-200 ${isEditing ? 'bg-white' : 'bg-gray-100 cursor-not-allowed'}`}
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
                            value={profile.phone}
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
                            value={profile.address}
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
                            value={profile.bio}
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
                            value={profile.tier}
                            readOnly // Tier is read-only, managed by admin
                            className="shadow appearance-none border rounded-md w-full py-3 px-4 text-gray-700 leading-tight focus:outline-none focus:ring-2 focus:ring-gray-300 bg-gray-100 cursor-not-allowed"
                        />
                    </div>
                </div>
            </div>

            <div className="flex justify-end space-x-4">
                {isEditing ? (
                    <>
                        <button
                            onClick={handleSave}
                            className="bg-purple-600 hover:bg-purple-700 text-white font-bold py-3 px-6 rounded-md shadow-lg transition duration-300 flex items-center"
                            disabled={loading}
                        >
                            {loading ? <Loader2 className="mr-2 h-5 w-5 animate-spin" /> : <Save className="mr-2" size={20} />} Save Changes
                        </button>
                        <button
                            onClick={handleCancel}
                            className="bg-gray-400 hover:bg-gray-500 text-white font-bold py-3 px-6 rounded-md shadow-lg transition duration-300 flex items-center"
                            disabled={loading}
                        >
                            <XCircle className="mr-2" size={20} /> Cancel
                        </button>
                    </>
                ) : (
                    <button
                        onClick={handleEditClick}
                        className="bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-3 px-6 rounded-md shadow-lg transition duration-300 flex items-center"
                    >
                        <User className="mr-2" size={20} /> Edit Profile
                    </button>
                )}
            </div>
        </div>
    );
};

export default UserProfile;

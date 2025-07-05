import React, { useState, useEffect } from 'react';
import { logEvent } from '../../utils/analytics_tracker'; // Adjust path as needed
import { User, Mail, Phone, MapPin, Save, XCircle } from 'lucide-react';

const UserProfile = ({ userId }) => {
    const [profile, setProfile] = useState({
        username: 'JohnDoe',
        email: 'john.doe@example.com',
        phone: '123-456-7890',
        address: '123 Main St, Anytown, USA',
        bio: 'A passionate user of the AI Assistant!',
        tier: 'Pro' // Example tier
    });
    const [isEditing, setIsEditing] = useState(false);
    const [statusMessage, setStatusMessage] = useState('');

    // Log page view when component mounts
    useEffect(() => {
        if (userId) {
            logEvent('page_view', { page_name: 'UserProfile', user_id: userId });
            console.log(`Analytics: Logged page_view for 'UserProfile' by user '${userId}'`);
        }
    }, [userId]); // Depend on userId to ensure it's available

    const handleChange = (e) => {
        const { name, value } = e.target;
        setProfile(prevProfile => ({
            ...prevProfile,
            [name]: value
        }));
    };

    const handleSave = () => {
        // Simulate API call to save profile
        console.log('Saving profile:', profile);
        setIsEditing(false);
        setStatusMessage('Profile saved successfully!');
        logEvent('ui_interaction', {
            component: 'UserProfile',
            action: 'Save Profile',
            details: { userId, changes: profile }, // Log details of the interaction
            user_id: userId,
            success: true
        });
        console.log(`Analytics: Logged ui_interaction 'Save Profile' by user '${userId}'`);

        // Clear status message after a few seconds
        setTimeout(() => setStatusMessage(''), 3000);
    };

    const handleCancel = () => {
        setIsEditing(false);
        setStatusMessage('Profile changes cancelled.');
        logEvent('ui_interaction', {
            component: 'UserProfile',
            action: 'Cancel Profile Edit',
            details: { userId },
            user_id: userId,
            success: true
        });
        console.log(`Analytics: Logged ui_interaction 'Cancel Profile Edit' by user '${userId}'`);
        setTimeout(() => setStatusMessage(''), 3000);
        // Optionally, reset profile to original state if you fetched it from a backend
    };

    const handleEditClick = () => {
        setIsEditing(true);
        logEvent('ui_interaction', {
            component: 'UserProfile',
            action: 'Edit Profile Click',
            details: { userId },
            user_id: userId,
            success: true
        });
        console.log(`Analytics: Logged ui_interaction 'Edit Profile Click' by user '${userId}'`);
    };

    return (
        <div className="p-6 bg-gray-100 min-h-screen font-inter text-gray-800 rounded-lg shadow-inner">
            <h1 className="text-4xl font-extrabold text-center text-purple-800 mb-8 flex items-center justify-center">
                <User className="mr-3 text-purple-600" size={36} />
                User Profile
            </h1>

            <div className="bg-white p-8 rounded-xl shadow-lg max-w-2xl mx-auto">
                {statusMessage && (
                    <div className="bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded-xl relative mb-6 shadow-md" role="alert">
                        <strong className="font-bold">Success!</strong>
                        <span className="block sm:inline"> {statusMessage}</span>
                    </div>
                )}

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">Username</label>
                        {isEditing ? (
                            <input
                                type="text"
                                name="username"
                                value={profile.username}
                                onChange={handleChange}
                                className="w-full p-3 border border-gray-300 rounded-md focus:ring-purple-500 focus:border-purple-500 shadow-sm"
                            />
                        ) : (
                            <p className="text-lg font-semibold text-gray-900 flex items-center">
                                <User size={18} className="mr-2 text-purple-500" /> {profile.username}
                            </p>
                        )}
                    </div>
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">Email</label>
                        {isEditing ? (
                            <input
                                type="email"
                                name="email"
                                value={profile.email}
                                onChange={handleChange}
                                className="w-full p-3 border border-gray-300 rounded-md focus:ring-purple-500 focus:border-purple-500 shadow-sm"
                            />
                        ) : (
                            <p className="text-lg font-semibold text-gray-900 flex items-center">
                                <Mail size={18} className="mr-2 text-purple-500" /> {profile.email}
                            </p>
                        )}
                    </div>
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">Phone</label>
                        {isEditing ? (
                            <input
                                type="text"
                                name="phone"
                                value={profile.phone}
                                onChange={handleChange}
                                className="w-full p-3 border border-gray-300 rounded-md focus:ring-purple-500 focus:border-purple-500 shadow-sm"
                            />
                        ) : (
                            <p className="text-lg font-semibold text-gray-900 flex items-center">
                                <Phone size={18} className="mr-2 text-purple-500" /> {profile.phone}
                            </p>
                        )}
                    </div>
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">Address</label>
                        {isEditing ? (
                            <input
                                type="text"
                                name="address"
                                value={profile.address}
                                onChange={handleChange}
                                className="w-full p-3 border border-gray-300 rounded-md focus:ring-purple-500 focus:border-purple-500 shadow-sm"
                            />
                        ) : (
                            <p className="text-lg font-semibold text-gray-900 flex items-center">
                                <MapPin size={18} className="mr-2 text-purple-500" /> {profile.address}
                            </p>
                        )}
                    </div>
                </div>

                <div className="mb-6">
                    <label className="block text-sm font-medium text-gray-700 mb-1">Bio</label>
                    {isEditing ? (
                        <textarea
                            name="bio"
                            value={profile.bio}
                            onChange={handleChange}
                            rows="4"
                            className="w-full p-3 border border-gray-300 rounded-md focus:ring-purple-500 focus:border-purple-500 shadow-sm"
                        ></textarea>
                    ) : (
                        <p className="text-lg text-gray-700 bg-gray-50 p-3 rounded-md border border-gray-200">{profile.bio}</p>
                    )}
                </div>

                <div className="mb-6">
                    <label className="block text-sm font-medium text-gray-700 mb-1">Subscription Tier</label>
                    <p className="text-lg font-bold text-purple-700 bg-purple-50 px-4 py-2 rounded-md inline-block shadow-sm">
                        {profile.tier}
                    </p>
                </div>

                <div className="flex justify-end space-x-4">
                    {isEditing ? (
                        <>
                            <button
                                onClick={handleSave}
                                className="bg-purple-600 hover:bg-purple-700 text-white font-bold py-3 px-6 rounded-md shadow-lg transition duration-300 flex items-center"
                            >
                                <Save className="mr-2" size={20} /> Save Changes
                            </button>
                            <button
                                onClick={handleCancel}
                                className="bg-gray-400 hover:bg-gray-500 text-white font-bold py-3 px-6 rounded-md shadow-lg transition duration-300 flex items-center"
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
        </div>
    );
};

export default UserProfile;

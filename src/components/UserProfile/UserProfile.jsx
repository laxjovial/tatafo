import React, { useState, useEffect, useCallback } from 'react';
// import { logEvent } from '../../services/analytics';
import './UserProfile.css';

import { FASTAPI_BASE_URL } from '../../config';

const UserProfile = ({ userId, auth }) => {
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
        if (!auth || !auth.currentUser) {
            setError("Authentication not ready. Please wait or log in again.");
            setLoading(false);
            return;
        }

        setLoading(true);
        setError(null);
        setMessage('');
        try {
            const idToken = await auth.currentUser?.getIdToken(true);
            if (!idToken) {
                throw new Error("No authentication token available. Please log in.");
            }

            const response = await fetch(`${FASTAPI_BASE_URL}/profile/${userId}`, {
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
            setUserData(data.user || data);
            // await logEvent('user_profile_view', { user_id: userId, status: 'success' });
        } catch (err) {
            console.error("Error fetching user profile:", err);
            setError(err.message);
            // await logEvent('user_profile_view', { user_id: userId, status: 'failure', error: err.message });
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

            const response = await fetch(`${FASTAPI_BASE_URL}/profile/update/${userId}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${idToken}`
                },
                body: JSON.stringify({
                    phone: userData.phone,
                    address: userData.address,
                    bio: userData.bio,
                    username: userData.username
                })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to update user profile.');
            }

            const data = await response.json();
            setMessage(data.message || 'Profile updated successfully!');
            setIsEditing(false);
            // await logEvent('user_profile_update', { user_id: userId, status: 'success' });
            fetchUserProfile();
        } catch (err) {
            console.error("Error updating user profile:", err);
            setError(err.message);
            // await logEvent('user_profile_update', { user_id: userId, status: 'failure', error: err.message });
        } finally {
            setLoading(false);
        }
    };

    const handleCancel = () => {
        setIsEditing(false);
        setMessage('Edit cancelled.');
        fetchUserProfile();
    };

    return (
        <div className="user-profile-container">
            <h2 className="profile-title">User Profile</h2>

            {message && (
                <div className={`message ${error ? 'error' : 'success'}`}>
                    {message}
                </div>
            )}

            <div className="profile-grid">
                <div className="profile-column">
                    <div className="form-group">
                        <label className="form-label" htmlFor="username">
                            Username
                        </label>
                        <input
                            type="text"
                            id="username"
                            name="username"
                            value={userData.username}
                            onChange={handleChange}
                            readOnly={!isEditing}
                            className={`form-input ${!isEditing ? 'readonly' : ''}`}
                        />
                    </div>
                    <div className="form-group">
                        <label className="form-label" htmlFor="email">
                            Email
                        </label>
                        <input
                            type="email"
                            id="email"
                            name="email"
                            value={userData.email}
                            readOnly
                            className="form-input readonly"
                        />
                    </div>
                    <div className="form-group">
                        <label className="form-label" htmlFor="phone">
                            Phone
                        </label>
                        <input
                            type="tel"
                            id="phone"
                            name="phone"
                            value={userData.phone}
                            onChange={handleChange}
                            readOnly={!isEditing}
                            className={`form-input ${!isEditing ? 'readonly' : ''}`}
                        />
                    </div>
                </div>
                <div className="profile-column">
                    <div className="form-group">
                        <label className="form-label" htmlFor="address">
                            Address
                        </label>
                        <input
                            type="text"
                            id="address"
                            name="address"
                            value={userData.address}
                            onChange={handleChange}
                            readOnly={!isEditing}
                            className={`form-input ${!isEditing ? 'readonly' : ''}`}
                        />
                    </div>
                    <div className="form-group">
                        <label className="form-label" htmlFor="bio">
                            Bio
                        </label>
                        <textarea
                            id="bio"
                            name="bio"
                            value={userData.bio}
                            onChange={handleChange}
                            readOnly={!isEditing}
                            rows="4"
                            className={`form-input ${!isEditing ? 'readonly' : ''}`}
                        ></textarea>
                    </div>
                    <div className="form-group">
                        <label className="form-label" htmlFor="tier">
                            Tier
                        </label>
                        <input
                            type="text"
                            id="tier"
                            name="tier"
                            value={userData.tier}
                            readOnly
                            className="form-input readonly"
                        />
                    </div>
                </div>
            </div>

            <div className="button-group">
                {loading ? (
                    <button className="button-disabled" disabled>
                        Processing...
                    </button>
                ) : !isEditing ? (
                    <button onClick={() => setIsEditing(true)} className="button-edit">
                        Edit Profile
                    </button>
                ) : (
                    <>
                        <button onClick={handleSave} className="button-save">
                            Save Changes
                        </button>
                        <button onClick={handleCancel} className="button-cancel">
                            Cancel
                        </button>
                    </>
                )}
            </div>
        </div>
    );
};

export default UserProfile;

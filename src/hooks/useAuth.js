import { useState, useEffect, useCallback } from 'react';
import { onAuthStateChanged, signOut } from 'firebase/auth';
import { auth } from '../services/firebase';
// import { logEvent } from '../services/analytics'; // To be re-enabled
import { FASTAPI_BASE_URL } from '../config';

export const useAuth = () => {
    const [user, setUser] = useState(null);
    const [userProfile, setUserProfile] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    const fetchUserProfile = useCallback(async (firebaseUser) => {
        if (!firebaseUser) {
            setUserProfile(null);
            return;
        }
        setLoading(true);
        setError(null);
        try {
            const idToken = await firebaseUser.getIdToken(true);
            const response = await fetch(`${FASTAPI_BASE_URL}/profile/${firebaseUser.uid}`, {
                headers: {
                    'Authorization': `Bearer ${idToken}`,
                },
            });

            if (!response.ok) {
                throw new Error('Failed to fetch user profile.');
            }
            const profileData = await response.json();
            setUserProfile(profileData);
            // await logEvent('user_profile_fetch', { uid: firebaseUser.uid }, true);
        } catch (err) {
            setError(err.message);
            setUserProfile(null);
            // await logEvent('user_profile_fetch', { uid: firebaseUser.uid, error: err.message }, false, err.message);
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        const unsubscribe = onAuthStateChanged(auth, (firebaseUser) => {
            setUser(firebaseUser);
            if (firebaseUser) {
                fetchUserProfile(firebaseUser);
            } else {
                setUserProfile(null);
                setLoading(false);
            }
        });

        return () => unsubscribe();
    }, [fetchUserProfile]);

    const logout = async () => {
        setLoading(true);
        try {
            await signOut(auth);
            setUser(null);
            setUserProfile(null);
            // await logEvent('user_logout', { uid: user.uid }, true);
        } catch (err) {
            setError(err.message);
            // await logEvent('user_logout', { uid: user.uid, error: err.message }, false, err.message);
        } finally {
            setLoading(false);
        }
    };

    const isAuthenticated = !!user;
    const isAdmin = userProfile?.roles?.includes('admin') ?? false;

    return {
        user,
        userProfile,
        isAuthenticated,
        isAdmin,
        loading,
        error,
        logout,
        refreshUserProfile: () => fetchUserProfile(user),
    };
};

export default useAuth;

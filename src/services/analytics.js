import { collection, doc, setDoc } from 'firebase/firestore';
import { db } from './firebase';

let authInstance = null;
let currentAppId = null;
let currentUserId = null;

export const initializeAnalytics = (auth, appId, userId) => {
    authInstance = auth;
    currentAppId = appId;
    currentUserId = userId;
    console.log(`Analytics initialized for app_id: ${currentAppId}, user_id: ${currentUserId}`);
};

export const logEvent = async (eventType, eventDetails, success = null, errorMessage = null) => {
    if (!db || !currentAppId || !currentUserId) {
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
        const analyticsCollectionRef = collection(db, `artifacts/${currentAppId}/public/data/analytics_events`);
        const newDocRef = doc(analyticsCollectionRef);
        await setDoc(newDocRef, eventData);
        console.log(`Analytics event '${eventType}' logged successfully for user ${currentUserId}.`);
    } catch (error) {
        console.error(`Error logging analytics event '${eventType}':`, error);
    }
};

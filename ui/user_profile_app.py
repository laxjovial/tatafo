# ui/user_profile_app.py

import streamlit as st
import requests
import logging
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# --- Configuration for FastAPI Backend ---
FASTAPI_BASE_URL = "http://localhost:8000" # Assuming FastAPI runs on port 8000

def get_user_profile_from_backend(user_token: str) -> Dict[str, Any]:
    """Fetches user profile from the backend."""
    try:
        response = requests.get(f"{FASTAPI_BASE_URL}/users/{user_token}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching user profile for {user_token}: {e}")
        st.error(f"Could not load user profile: {e}. Please ensure the backend is running.")
        return {"user_id": user_token, "username": "Error", "tier": "Unknown", "roles": ["error"],
                "subscription_start_date": "N/A", "subscription_end_date": "N/A",
                "days_left": "N/A", "next_subscription_date": "N/A"} # Fallback

def app():
    st.title("👤 User Profile")

    if "user_token" not in st.session_state or not st.session_state.user_token:
        st.warning("Please log in to view your profile.")
        return

    user_token = st.session_state.user_token
    user_profile = get_user_profile_from_backend(user_token)

    if user_profile:
        st.header(f"Welcome, {user_profile.get('username', 'User')}!")
        st.markdown("---")

        st.subheader("Account Information")
        st.write(f"**User ID:** `{user_profile.get('user_id', 'N/A')}`")
        st.write(f"**Email:** {user_profile.get('email', 'N/A')}")
        st.write(f"**Tier:** {user_profile.get('tier', 'N/A').capitalize()}")
        st.write(f"**Roles:** {', '.join(user_profile.get('roles', []))}")

        st.markdown("---")
        st.subheader("Subscription Details")
        
        sub_start = user_profile.get('subscription_start_date', 'N/A')
        sub_end = user_profile.get('subscription_end_date', 'N/A')
        days_left = user_profile.get('days_left', 'N/A')
        next_sub_date = user_profile.get('next_subscription_date', 'N/A')

        st.write(f"**Subscription Start Date:** {sub_start}")
        st.write(f"**Subscription End Date:** {sub_end}")
        st.write(f"**Days Remaining:** {days_left}")
        st.write(f"**Next Subscription Date:** {next_sub_date}")

        if user_profile.get('tier') == 'free':
            st.info("You are currently on the Free tier. Upgrade to unlock more features and extended subscriptions!")
        elif days_left != "N/A" and isinstance(days_left, int) and days_left <= 7:
            st.warning(f"Your subscription is ending in {days_left} day(s)! Renew now to continue enjoying all features.")
        elif days_left == 0 and user_profile.get('tier') != 'free':
            st.error("Your subscription has expired! Please renew to regain access to premium features.")
        
        st.markdown("---")
        st.subheader("Manage Account")
        # These buttons would navigate to other pages in a multi-page app
        if st.button("Change Password"):
            st.session_state.current_page = "Change Password" # Assuming main_app.py handles navigation
            st.rerun()
        if st.button("Manage Subscription"):
            st.info("This would navigate to a subscription management page (future feature).")

    else:
        st.error("Failed to load user profile. Please try again or contact support.")

# Example of how to run this app standalone for testing
if __name__ == "__main__":
    # Mock session state for standalone testing
    if "user_token" not in st.session_state:
        st.session_state.user_token = "mock_premium_token" # Or "mock_free_token", etc.
    
    # Mock requests.get for backend calls if running standalone without FastAPI
    import unittest.mock as mock
    original_requests_get = requests.get

    def mock_requests_get(url, *args, **kwargs):
        if f"{FASTAPI_BASE_URL}/users/" in url:
            user_id = url.split('/')[-1]
            mock_users_data = {
                "mock_free_token": {
                    "user_id": "mock_free_token", "username": "FreeUser", "email": "free@example.com", "tier": "free", "roles": ["user"],
                    "subscription_start_date": "N/A", "subscription_end_date": "N/A", "days_left": "N/A", "next_subscription_date": "N/A"
                },
                "mock_basic_token": {
                    "user_id": "mock_basic_token", "username": "BasicUser", "email": "basic@example.com", "tier": "basic", "roles": ["user"],
                    "subscription_start_date": (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d"),
                    "subscription_end_date": (datetime.now() + timedelta(days=20)).strftime("%Y-%m-%d"),
                    "days_left": 20,
                    "next_subscription_date": (datetime.now() + timedelta(days=21)).strftime("%Y-%m-%d")
                },
                "mock_premium_token": {
                    "user_id": "mock_premium_token", "username": "PremiumUser", "email": "premium@example.com", "tier": "premium", "roles": ["user"],
                    "subscription_start_date": (datetime.now() - timedelta(days=15)).strftime("%Y-%m-%d"),
                    "subscription_end_date": (datetime.now() + timedelta(days=45)).strftime("%Y-%m-%d"),
                    "days_left": 45,
                    "next_subscription_date": (datetime.now() + timedelta(days=46)).strftime("%Y-%m-%d")
                }
            }
            user_data = mock_users_data.get(user_id)
            if user_data:
                mock_response = mock.Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = user_data
                mock_response.raise_for_status = lambda: None
                return mock_response
        return original_requests_get(url, *args, **kwargs)

    requests.get = mock_requests_get
    
    app()

    # Restore original requests.get after testing
    requests.get = original_requests_get

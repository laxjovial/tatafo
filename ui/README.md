 ui/main_app.py (Updated to Use Tier Hierarchy from user_manager)
ui/main_app.py (Updated for Tier Hierarchy)
Jul 2, 7:19 AM

Open

Key Changes in ui/main_app.py:

_TIER_HIERARCHY Import: Removed the hardcoded TIER_HIERARCHY dictionary and now imports _TIER_HIERARCHY directly from utils.user_manager. This ensures that the tier levels are consistently defined in data/tiers.yaml and loaded by user_manager.

has_access Function: The has_access function now explicitly uses the imported _TIER_HIERARCHY.

New Mini Chatbot Page: Added "Mini Chatbot": {"app": mini_chatbot_app, "tier_access": "user", "roles": ["user", "admin"]} to the PAGES dictionary. This page will be accessible to all registered users.

initialize_app_config(): Added a mock for st.secrets.firebase_config to prevent errors during local testing if Firebase config isn't explicitly set in secrets.toml.

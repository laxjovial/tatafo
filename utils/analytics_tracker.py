# utils/analytics_tracker.py

import logging
from datetime import datetime
from typing import Optional, Dict, Any

# Firebase imports (assuming these are globally available or handled by the main app)
# We will use the global __app_id, __firebase_config, __initial_auth_token
# and firebase client libraries for Firestore operations.

logger = logging.getLogger(__name__)

# Global Firestore and Auth instances will be initialized in the main app
# and passed or accessed via global scope in a Streamlit context.
# For standalone testing, we'll mock them.
db = None
auth = None
app_id = None
user_id = None # This will be set by the main app's auth state

def initialize_analytics(firestore_db, firebase_auth, current_app_id: str, current_user_id: str):
    """
    Initializes the analytics module with Firestore and Auth instances.
    This should be called once at the application startup.
    """
    global db, auth, app_id, user_id
    db = firestore_db
    auth = firebase_auth
    app_id = current_app_id
    user_id = current_user_id
    logger.info(f"Analytics initialized for app_id: {app_id}, user_id: {user_id}")

async def log_event(
    event_type: str,
    event_details: Dict[str, Any],
    user_token: Optional[str] = None,
    success: Optional[bool] = None,
    error_message: Optional[str] = None
):
    """
    Logs an analytics event to Firestore.

    Args:
        event_type (str): The type of event (e.g., "tool_usage", "query_failure", "user_login").
        event_details (Dict[str, Any]): A dictionary containing specific details about the event.
        user_token (str, optional): The user's authentication token.
        success (bool, optional): Whether the event was successful.
        error_message (str, optional): An error message if the event failed.
    """
    if db is None or app_id is None:
        logger.warning("Analytics not initialized. Cannot log event.")
        return

    # Ensure user_id is available; use a default or anonymous if not authenticated
    current_user_id = user_id if user_id else (auth.currentUser.uid if auth and auth.currentUser else "anonymous")

    event_data = {
        "timestamp": datetime.now().isoformat(),
        "app_id": app_id,
        "user_id": current_user_id,
        "event_type": event_type,
        "details": event_details,
        "success": success,
        "error_message": error_message
    }

    try:
        # Store analytics in a public collection for easier aggregation/reporting
        # Path: /artifacts/{appId}/public/analytics_logs/{docId}
        collection_path = f"artifacts/{app_id}/public/analytics_logs"
        
        from firebase_admin import firestore # Import locally for CLI test mock
        await db.collection(collection_path).add(event_data)
        logger.info(f"Logged analytics event: {event_type} for user {current_user_id}")
    except Exception as e:
        logger.error(f"Failed to log analytics event to Firestore: {e}", exc_info=True)

async def log_tool_usage(
    tool_name: str,
    tool_params: Dict[str, Any],
    user_token: Optional[str] = None,
    success: bool = True,
    error_message: Optional[str] = None
):
    """
    Logs the usage of a specific tool.
    """
    event_details = {
        "tool_name": tool_name,
        "tool_params": tool_params
    }
    await log_event("tool_usage", event_details, user_token, success, error_message)

async def log_query_failure(
    query: str,
    reason: str,
    user_token: Optional[str] = None,
    tool_attempted: Optional[str] = None
):
    """
    Logs a query failure event.
    """
    event_details = {
        "query": query,
        "reason": reason,
        "tool_attempted": tool_attempted
    }
    await log_event("query_failure", event_details, user_token, success=False)

# CLI Test (optional)
if __name__ == "__main__":
    import asyncio
    from unittest.mock import MagicMock, AsyncMock, patch
    import sys

    logging.basicConfig(level=logging.INFO)

    # Mock Firestore and Auth for testing
    mock_db = MagicMock()
    mock_auth = MagicMock()
    mock_auth.currentUser = MagicMock(uid="mock_user_123")
    
    # Mock the add method to be an async mock
    mock_db.collection.return_value.add = AsyncMock(return_value=MagicMock(id="mock_doc_id"))

    # Patch firebase_admin.firestore for the local import within log_event
    with patch.dict(sys.modules, {'firebase_admin.firestore': MagicMock(firestore=MagicMock())}):
        sys.modules['firebase_admin.firestore'].firestore.CollectionReference = MagicMock()
        sys.modules['firebase_admin.firestore'].firestore.DocumentReference = MagicMock()

        async def run_tests():
            print("--- Initializing Analytics ---")
            initialize_analytics(mock_db, mock_auth, "test_app_id", "mock_user_123")

            print("\n--- Testing log_tool_usage (Success) ---")
            await log_tool_usage(
                tool_name="get_stock_price",
                tool_params={"symbol": "AAPL"},
                user_token="mock_user_token_pro",
                success=True
            )
            mock_db.collection.return_value.add.assert_called_once()
            args, kwargs = mock_db.collection.return_value.add.call_args
            logged_data = args[0]
            print(f"Logged Data: {logged_data}")
            assert logged_data["event_type"] == "tool_usage"
            assert logged_data["details"]["tool_name"] == "get_stock_price"
            assert logged_data["success"] is True
            mock_db.collection.return_value.add.reset_mock() # Reset mock for next test

            print("\n--- Testing log_tool_usage (Failure) ---")
            await log_tool_usage(
                tool_name="search_flights",
                tool_params={"origin": "XYZ"},
                user_token="mock_user_token_free",
                success=False,
                error_message="Invalid origin code"
            )
            mock_db.collection.return_value.add.assert_called_once()
            args, kwargs = mock_db.collection.return_value.add.call_args
            logged_data = args[0]
            print(f"Logged Data: {logged_data}")
            assert logged_data["event_type"] == "tool_usage"
            assert logged_data["details"]["tool_name"] == "search_flights"
            assert logged_data["success"] is False
            assert "Invalid origin code" in logged_data["error_message"]
            mock_db.collection.return_value.add.reset_mock()

            print("\n--- Testing log_query_failure ---")
            await log_query_failure(
                query="What is the meaning of life?",
                reason="No tool available for philosophical queries.",
                user_token="mock_user_token_pro",
                tool_attempted=None
            )
            mock_db.collection.return_value.add.assert_called_once()
            args, kwargs = mock_db.collection.return_value.add.call_args
            logged_data = args[0]
            print(f"Logged Data: {logged_data}")
            assert logged_data["event_type"] == "query_failure"
            assert logged_data["details"]["query"] == "What is the meaning of life?"
            assert logged_data["success"] is False
            assert "No tool available" in logged_data["details"]["reason"]
            mock_db.collection.return_value.add.reset_mock()

            print("\nAll analytics tests completed.")

        asyncio.run(run_tests())

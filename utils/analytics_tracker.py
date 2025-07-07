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
    user_id: Optional[str] = None, # Renamed from user_token to user_id for clarity
    success: Optional[bool] = None,
    error_message: Optional[str] = None,
    log_from_backend: bool = False # New parameter to indicate if log is from backend
):
    """
    Logs an analytics event to Firestore.

    Args:
        event_type (str): The type of event (e.g., "tool_usage", "query_failure", "user_login").
        event_details (Dict[str, Any]): A dictionary containing specific details about the event.
        user_id (str, optional): The user's unique ID.
        success (bool, optional): Whether the event was successful.
        error_message (str, optional): An error message if the event failed.
        log_from_backend (bool): True if the log is initiated from the backend, False if from frontend.
    """
    if db is None or app_id is None:
        logger.warning("Analytics not initialized. Cannot log event.")
        return

    # Determine the user ID for logging
    # If user_id is explicitly passed, use it. Otherwise, try global user_id or anonymous.
    current_user_id = user_id if user_id else (globals().get('user_id') or (auth.currentUser.uid if auth and auth.currentUser else "anonymous"))

    event_data = {
        "timestamp": datetime.now().isoformat(),
        "app_id": app_id,
        "user_id": current_user_id,
        "event_type": event_type,
        "details": event_details,
        "success": success,
        "error_message": error_message,
        "log_from_backend": log_from_backend # Store the origin of the log
    }

    try:
        # Store analytics in a public collection for easier aggregation/reporting
        # Path: /artifacts/{appId}/public/analytics_logs/{docId}
        collection_path = f"artifacts/{app_id}/public/analytics_logs"
        
        # Import firestore locally to avoid circular dependencies if analytics_tracker
        # is imported by modules that initialize firestore.
        from firebase_admin import firestore 
        await db.collection(collection_path).add(event_data)
        logger.info(f"Logged analytics event: {event_type} for user {current_user_id}")
    except Exception as e:
        logger.error(f"Failed to log analytics event to Firestore: {e}", exc_info=True)

async def log_tool_usage(
    tool_name: str,
    tool_params: Dict[str, Any],
    user_id: Optional[str] = None, # Renamed from user_token
    success: bool = True,
    error_message: Optional[str] = None,
    log_from_backend: bool = False
):
    """
    Logs the usage of a specific tool.
    """
    event_details = {
        "tool_name": tool_name,
        "tool_params": tool_params
    }
    await log_event("tool_usage", event_details, user_id, success, error_message, log_from_backend)

async def log_query_failure(
    query: str,
    reason: str,
    user_id: Optional[str] = None, # Renamed from user_token
    tool_attempted: Optional[str] = None,
    log_from_backend: bool = False
):
    """
    Logs a query failure event.
    """
    event_details = {
        "query": query,
        "reason": reason,
        "tool_attempted": tool_attempted
    }
    await log_event("query_failure", event_details, user_id, success=False, log_from_backend=log_from_backend)

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
                user_id="mock_user_token_pro",
                success=True,
                log_from_backend=True
            )
            mock_db.collection.return_value.add.assert_called_once()
            args, kwargs = mock_db.collection.return_value.add.call_args
            logged_data = args[0]
            print(f"Logged Data: {logged_data}")
            assert logged_data["event_type"] == "tool_usage"
            assert logged_data["details"]["tool_name"] == "get_stock_price"
            assert logged_data["success"] is True
            assert logged_data["log_from_backend"] is True
            mock_db.collection.return_value.add.reset_mock() # Reset mock for next test

            print("\n--- Testing log_tool_usage (Failure) ---")
            await log_tool_usage(
                tool_name="search_flights",
                tool_params={"origin": "XYZ"},
                user_id="mock_user_token_free",
                success=False,
                error_message="Invalid origin code",
                log_from_backend=True
            )
            mock_db.collection.return_value.add.assert_called_once()
            args, kwargs = mock_db.collection.return_value.add.call_args
            logged_data = args[0]
            print(f"Logged Data: {logged_data}")
            assert logged_data["event_type"] == "tool_usage"
            assert logged_data["details"]["tool_name"] == "search_flights"
            assert logged_data["success"] is False
            assert "Invalid origin code" in logged_data["error_message"]
            assert logged_data["log_from_backend"] is True
            mock_db.collection.return_value.add.reset_mock()

            print("\n--- Testing log_query_failure ---")
            await log_query_failure(
                query="What is the meaning of life?",
                reason="No tool available for philosophical queries.",
                user_id="mock_user_token_pro",
                log_from_backend=True
            )
            mock_db.collection.return_value.add.assert_called_once()
            args, kwargs = mock_db.collection.return_value.add.call_args
            logged_data = args[0]
            print(f"Logged Data: {logged_data}")
            assert logged_data["event_type"] == "query_failure"
            assert logged_data["details"]["query"] == "What is the meaning of life?"
            assert logged_data["success"] is False
            assert "No tool available" in logged_data["details"]["reason"]
            assert logged_data["log_from_backend"] is True
            mock_db.collection.return_value.add.reset_mock()

            print("\nAll analytics tests completed.")

        asyncio.run(run_tests())

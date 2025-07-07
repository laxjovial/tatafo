# utils/analytics_tracker.py

import logging
from datetime import datetime, timezone # Import timezone
from typing import Optional, Dict, Any
import asyncio

logger = logging.getLogger(__name__)

# Global variables for Firebase instances and app_id, initialized by the main app
_db = None
_auth = None
_app_id = None
_backend_user_id = None # To distinguish backend-initiated logs

def initialize_analytics(db_instance, auth_instance, app_id: str, backend_user_id: str = "backend-service"):
    """
    Initializes the analytics tracker with Firebase Firestore and Auth instances.
    This should be called once at application startup.
    """
    global _db, _auth, _app_id, _backend_user_id
    _db = db_instance
    _auth = auth_instance
    _app_id = app_id
    _backend_user_id = backend_user_id
    logger.info(f"Analytics Tracker initialized for app_id: {_app_id}")

async def log_event(
    event_type: str,
    details: Dict[str, Any],
    user_id: Optional[str] = None, # Corrected: user_id is now a direct parameter
    success: bool = True, # Corrected: success is now a direct parameter
    error_message: Optional[str] = None, # Corrected: error_message is now a direct parameter
    log_from_backend: bool = True # New parameter to indicate if log is from backend
):
    """
    Logs an analytics event to Firestore.
    
    Args:
        event_type (str): The type of event (e.g., 'page_view', 'tool_usage', 'user_login').
        details (Dict[str, Any]): A dictionary containing specific details about the event.
        user_id (Optional[str]): The ID of the user performing the action.
                                  Defaults to the backend service user ID if not provided and from_backend is True.
        success (bool): Whether the operation was successful.
        error_message (Optional[str]): An error message if the operation failed.
        log_from_backend (bool): True if the event is logged from the backend, False if from frontend.
    """
    if _db is None:
        logger.warning("Firestore DB not initialized for analytics. Event not logged.")
        return

    # Determine the collection path based on where the log originated
    if log_from_backend:
        # Backend-initiated logs (e.g., tool usage, internal processes)
        # Use a fixed backend user ID or the provided user_id
        effective_user_id = user_id if user_id else _backend_user_id
        collection_path = f"artifacts/{_app_id}/backend_analytics/{effective_user_id}/events"
    else:
        # Frontend-initiated logs (sent via /log-frontend-analytics endpoint)
        # These logs already contain the user_id from the frontend
        if not user_id:
            logger.error("Frontend analytics event received without user_id. Event not logged.")
            return
        collection_path = f"artifacts/{_app_id}/frontend_analytics/{user_id}/events"

    event_data = {
        "event_type": event_type,
        "timestamp": datetime.now(timezone.utc).isoformat(), # Use timezone.utc and isoformat
        "details": details,
        "user_id": user_id, # Store the user_id explicitly
        "success": success,
        "error_message": error_message,
        "source": "backend" if log_from_backend else "frontend"
    }

    try:
        # Firestore collection path must be odd number of elements.
        # artifacts/{app_id}/backend_analytics/{user_id}/events (5 elements, odd)
        # artifacts/{app_id}/frontend_analytics/{user_id}/events (5 elements, odd)
        await _db.collection(collection_path).add(event_data)
        logger.debug(f"Analytics event '{event_type}' logged successfully to {collection_path}")
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
    details = {
        "tool_name": tool_name,
        "tool_params": tool_params,
    }
    await log_event(
        event_type="tool_usage",
        details=details,
        user_id=user_token, # user_token is the user_id here
        success=success,
        error_message=error_message,
        log_from_backend=True # Tool usage is always from backend
    )

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
    await log_event("query_failure", event_details, user_id=user_token, success=False, log_from_backend=True)

# CLI Test (optional) - This part remains the same for testing purposes
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
                query="What is the meaning of life?"),
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

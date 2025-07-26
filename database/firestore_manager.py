# database/firestore_manager.py

import logging
from typing import Dict, Any, Optional, List
import asyncio
from datetime import datetime, timezone

# Firebase imports (assuming these are globally available or handled by the main app)
# We will use the global firebase_admin.firestore for operations.
import firebase_admin
from firebase_admin import firestore
from firebase_admin import auth as firebase_auth # Import auth specifically

logger = logging.getLogger(__name__)

class FirestoreManager:
    """
    Manages interactions with Google Cloud Firestore.
    Provides methods for common database operations.
    """
    def __init__(self, db_instance=None, auth_instance=None):
        """
        Initializes the FirestoreManager.
        Args:
            db_instance: The initialized Firebase Firestore client. If None, it attempts to get it.
            auth_instance: The initialized Firebase Auth client. If None, it attempts to get it.
        """
        self._db = db_instance if db_instance else firestore.client()
        self._auth = auth_instance if auth_instance else firebase_auth
        logger.info("FirestoreManager initialized.")

    async def get_doc(self, collection_path: str, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a single document from Firestore.

        Args:
            collection_path (str): The path to the collection (e.g., "users").
            doc_id (str): The ID of the document to retrieve.

        Returns:
            Optional[Dict[str, Any]]: The document data as a dictionary, or None if not found.
        """
        try:
            doc_ref = self._db.collection(collection_path).document(doc_id)
            doc = await asyncio.to_thread(doc_ref.get)
            if doc.exists:
                logger.debug(f"Document retrieved: {collection_path}/{doc_id}")
                return doc.to_dict()
            else:
                logger.info(f"Document not found: {collection_path}/{doc_id}")
                return None
        except Exception as e:
            logger.error(f"Error getting document {collection_path}/{doc_id}: {e}", exc_info=True)
            return None

    async def set_doc(self, collection_path: str, doc_id: str, data: Dict[str, Any], merge: bool = False) -> bool:
        """
        Sets (creates or overwrites) a document in Firestore.
        If merge is True, it merges the data with existing document data.

        Args:
            collection_path (str): The path to the collection (e.g., "users").
            doc_id (str): The ID of the document to set.
            data (Dict[str, Any]): The data to set.
            merge (bool): If True, merges the data with existing document.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            doc_ref = self._db.collection(collection_path).document(doc_id)
            await asyncio.to_thread(doc_ref.set, data, merge=merge)
            logger.info(f"Document set/updated: {collection_path}/{doc_id} (merge={merge})")
            return True
        except Exception as e:
            logger.error(f"Error setting document {collection_path}/{doc_id}: {e}", exc_info=True)
            return False

    async def update_doc(self, collection_path: str, doc_id: str, data: Dict[str, Any]) -> bool:
        """
        Updates an existing document in Firestore.

        Args:
            collection_path (str): The path to the collection (e.g., "users").
            doc_id (str): The ID of the document to update.
            data (Dict[str, Any]): The data to update.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            doc_ref = self._db.collection(collection_path).document(doc_id)
            await asyncio.to_thread(doc_ref.update, data)
            logger.info(f"Document updated: {collection_path}/{doc_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating document {collection_path}/{doc_id}: {e}", exc_info=True)
            return False

    async def add_doc(self, collection_path: str, data: Dict[str, Any]) -> Optional[str]:
        """
        Adds a new document to a collection with an auto-generated ID.

        Args:
            collection_path (str): The path to the collection (e.g., "analytics_logs").
            data (Dict[str, Any]): The data to add.

        Returns:
            Optional[str]: The ID of the newly created document, or None if unsuccessful.
        """
        try:
            collection_ref = self._db.collection(collection_path)
            update_time, doc_ref = await asyncio.to_thread(collection_ref.add, data)
            logger.info(f"Document added to {collection_path} with ID: {doc_ref.id}")
            return doc_ref.id
        except Exception as e:
            logger.error(f"Error adding document to {collection_path}: {e}", exc_info=True)
            return None

    async def delete_doc(self, collection_path: str, doc_id: str) -> bool:
        """
        Deletes a document from Firestore.

        Args:
            collection_path (str): The path to the collection (e.g., "users").
            doc_id (str): The ID of the document to delete.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            doc_ref = self._db.collection(collection_path).document(doc_id)
            await asyncio.to_thread(doc_ref.delete)
            logger.info(f"Document deleted: {collection_path}/{doc_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting document {collection_path}/{doc_id}: {e}", exc_info=True)
            return False

    async def get_analytics_events(
        self,
        event_type: Optional[str] = None,
        user_id: Optional[str] = None,
        start_date: Optional[str] = None, # YYYY-MM-DD
        end_date: Optional[str] = None,   # YYYY-MM-DD
    ) -> List[Dict[str, Any]]:
        """
        Retrieves analytics events from Firestore with optional filters.
        """
        events_ref = self._db.collection(f"artifacts/{firebase_admin.get_app().name}/public/data/analytics_logs")
        query = events_ref

        if event_type:
            query = query.where("event_type", "==", event_type)
        if user_id:
            query = query.where("user_id", "==", user_id)

        # Date filtering
        if start_date:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            query = query.where("timestamp", ">=", start_dt.isoformat())
        if end_date:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59, tzinfo=timezone.utc)
            query = query.where("timestamp", "<=", end_dt.isoformat())

        # Note: Firestore queries with range filters on different fields or
        # range filters on a field and an equality filter on another field
        # often require composite indexes. If you encounter errors, you might
        # need to create these indexes in the Firebase console.
        # For simplicity, we're not adding orderBy here to avoid index issues,
        # but in a real app, you'd likely want to order by timestamp.

        try:
            docs = await asyncio.to_thread(query.stream)
            events = []
            for doc in docs:
                event_data = doc.to_dict()
                events.append(event_data)
            logger.info(f"Retrieved {len(events)} analytics events.")
            return events
        except Exception as e:
            logger.error(f"Error retrieving analytics events: {e}", exc_info=True)
            return []

# CLI Test (optional)
if __name__ == "__main__":
    from unittest.mock import MagicMock, patch
    import os
    import json

    # Mock Firebase Admin SDK initialization for CLI test
    # This part is crucial for making the FirestoreManager testable outside FastAPI
    if not firebase_admin._apps:
        # Create a dummy credential object for local testing
        dummy_cred_path = "/tmp/dummy_firebase_admin_cert.json"
        dummy_cred_content = {
            "type": "service_account",
            "project_id": "test-project",
            "private_key_id": "dummy_key_id",
            "private_key": "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n",
            "client_email": "dummy@test-project.iam.gserviceaccount.com",
            "client_id": "dummy_client_id",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/dummy.iam.gserviceaccount.com",
            "universe_domain": "googleapis.com"
        }
        with open(dummy_cred_path, "w") as f:
            json.dump(dummy_cred_content, f)
        os.environ["FIREBASE_ADMIN_CERT"] = json.dumps(dummy_cred_content) # Set env var for initialization

        try:
            firebase_admin.initialize_app(firebase_admin.credentials.Certificate(dummy_cred_path), name="test-app")
            logger.info("Firebase Admin SDK initialized for CLI test.")
        except ValueError as e:
            logger.warning(f"Firebase Admin SDK already initialized or error: {e}")
        finally:
            # Clean up dummy cred file
            if os.path.exists(dummy_cred_path):
                os.remove(dummy_cred_path)

    async def run_tests():
        # Instantiate FirestoreManager (it will use the initialized firebase_admin db)
        # Pass the initialized auth instance as well
        manager = FirestoreManager(auth_instance=firebase_admin.auth) 

        test_collection = "test_collection"
        test_doc_id = "test_doc"
        test_data = {"name": "Test User", "age": 30, "timestamp": datetime.now(timezone.utc).isoformat()}
        updated_data = {"age": 31, "city": "New York"}
        
        # Test set_doc (create)
        print("\n--- Testing set_doc (create) ---")
        success = await manager.set_doc(test_collection, test_doc_id, test_data)
        print(f"set_doc (create) successful: {success}")
        assert success

        # Test get_doc
        print("\n--- Testing get_doc ---")
        retrieved_data = await manager.get_doc(test_collection, test_doc_id)
        print(f"get_doc data: {retrieved_data}")
        assert retrieved_data is not None
        assert retrieved_data["name"] == test_data["name"]

        # Test update_doc
        print("\n--- Testing update_doc ---")
        success = await manager.update_doc(test_collection, test_doc_id, updated_data)
        print(f"update_doc successful: {success}")
        assert success
        retrieved_data = await manager.get_doc(test_collection, test_doc_id)
        print(f"get_doc after update: {retrieved_data}")
        assert retrieved_data["age"] == updated_data["age"]
        assert retrieved_data["city"] == updated_data["city"]
        assert retrieved_data["name"] == test_data["name"] # Should still be there due to update behavior

        # Test add_doc
        print("\n--- Testing add_doc ---")
        new_doc_id = await manager.add_doc(test_collection, {"item": "new item", "value": 100})
        print(f"add_doc new ID: {new_doc_id}")
        assert new_doc_id is not None
        new_item_data = await manager.get_doc(test_collection, new_doc_id)
        assert new_item_data["item"] == "new item"

        # Test get_analytics_events (basic)
        print("\n--- Testing get_analytics_events ---")
        # Log a dummy analytics event for testing retrieval
        dummy_event_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "app_id": firebase_admin.get_app("test-app").name, # Use the test app's name
            "user_id": "test_user_analytics",
            "event_type": "test_event",
            "details": {"key": "value"},
            "success": True,
            "error_message": None,
            "log_from_backend": True
        }
        await manager.add_doc(f"artifacts/{firebase_admin.get_app('test-app').name}/public/data/analytics_logs", dummy_event_data)
        
        events = await manager.get_analytics_events(event_type="test_event", user_id="test_user_analytics")
        print(f"Retrieved analytics events: {events}")
        assert len(events) >= 1
        assert any(e["event_type"] == "test_event" for e in events)

        # Test delete_doc
        print("\n--- Testing delete_doc ---")
        success = await manager.delete_doc(test_collection, test_doc_id)
        print(f"delete_doc successful: {success}")
        assert success
        retrieved_data = await manager.get_doc(test_collection, test_doc_id)
        assert retrieved_data is None

        # Clean up the added doc
        if new_doc_id:
            await manager.delete_doc(test_collection, new_doc_id)

        print("\nAll FirestoreManager tests completed.")

    if __name__ == "__main__":
        asyncio.run(run_tests())

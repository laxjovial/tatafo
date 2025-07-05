# database/firestore_manager.py

import firebase_admin
from firebase_admin import firestore, auth # No need for credentials here anymore
import logging
import json # Not strictly needed here anymore, but good to keep if other parts use it
from typing import Optional, Dict, Any, List
from datetime import datetime

# Assuming ConfigManager is accessible, but it won't be used for Admin SDK init here
from config.config_manager import config_manager

logger = logging.getLogger(__name__)

class FirestoreManager:
    _instance = None
    _db = None
    _auth = None
    _app_id = None # Store the app_id here (optional, for context)
    _is_initialized = False

    # The constructor now takes initialized db and auth instances
    def __new__(cls, db_instance=None, auth_instance=None):
        if cls._instance is None:
            cls._instance = super(FirestoreManager, cls).__new__(cls)
            # Initialize only if instances are provided and not already initialized
            if db_instance and auth_instance and not cls._instance._is_initialized:
                cls._instance._db = db_instance
                cls._instance._auth = auth_instance
                # Optionally, get app_id from the initialized Firebase app
                try:
                    cls._instance._app_id = firebase_admin.get_app().project_id
                except ValueError:
                    # Fallback if app_id cannot be retrieved from initialized app (e.g., mock env)
                    # This fallback uses the client config projectId, which is fine for app_id.
                    firebase_client_config_str = config_manager.get_secret('firebase_config')
                    if firebase_client_config_str:
                        try:
                            firebase_client_config = json.loads(firebase_client_config_str)
                            cls._instance._app_id = firebase_client_config.get('projectId', 'default-app-id')
                        except json.JSONDecodeError:
                            cls._instance._app_id = 'default-app-id'
                            logger.warning("Failed to parse firebase_config from secrets.toml for app_id fallback.")
                    else:
                        cls._instance._app_id = 'default-app-id'
                        logger.warning("Firebase client config not found for app_id fallback.")

                cls._instance._is_initialized = True
                logger.info("Firestore client and Auth received and initialized in FirestoreManager.")
            elif not db_instance or not auth_instance:
                # This case should ideally not happen if called correctly from main.py
                logger.error("FirestoreManager instantiated without providing initialized db and auth instances.")
                # For robustness, we could raise an error here, but for now, just log.
                # raise RuntimeError("FirestoreManager must be initialized with db and auth instances.")
        return cls._instance

    @property
    def db(self):
        if not self._is_initialized:
            raise RuntimeError("Firestore not initialized. Ensure FirestoreManager is instantiated with db and auth.")
        return self._db

    @property
    def auth(self):
        if not self._is_initialized:
            raise RuntimeError("Firebase Auth not initialized. Ensure FirestoreManager is instantiated with db and auth.")
        return self._auth

    @property
    def app_id(self):
        return self._app_id

    # --- Utility methods for Firestore operations (rest of the class remains the same) ---

    def get_document(self, collection_path: str, document_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single document."""
        try:
            doc_ref = self.db.collection(collection_path).document(document_id)
            doc = doc_ref.get()
            if doc.exists:
                logger.debug(f"Document {document_id} found in {collection_path}")
                return doc.to_dict()
            else:
                logger.debug(f"Document {document_id} not found in {collection_path}")
                return None
        except Exception as e:
            logger.error(f"Error getting document {document_id} from {collection_path}: {e}")
            return None

    def add_document(self, collection_path: str, data: Dict[str, Any]) -> str:
        """Adds a new document to a collection with an auto-generated ID."""
        try:
            doc_ref = self.db.collection(collection_path).add(data)
            logger.info(f"Document added to {collection_path} with ID: {doc_ref[1].id}")
            return doc_ref[1].id
        except Exception as e:
            logger.error(f"Error adding document to {collection_path}: {e}")
            raise

    def set_document(self, collection_path: str, document_id: str, data: Dict[str, Any], merge: bool = False):
        """Sets a document with a specific ID, optionally merging data."""
        try:
            doc_ref = self.db.collection(collection_path).document(document_id)
            doc_ref.set(data, merge=merge)
            logger.info(f"Document {document_id} set in {collection_path} (merge={merge})")
        except Exception as e:
            logger.error(f"Error setting document {document_id} in {collection_path}: {e}")
            raise

    def update_document(self, collection_path: str, document_id: str, data: Dict[str, Any]):
        """Updates specific fields of a document."""
        try:
            doc_ref = self.db.collection(collection_path).document(document_id)
            doc_ref.update(data)
            logger.info(f"Document {document_id} updated in {collection_path}")
        except Exception as e:
            logger.error(f"Error updating document {document_id} in {collection_path}: {e}")
            raise

    def delete_document(self, collection_path: str, document_id: str):
        """Deletes a document."""
        try:
            self.db.collection(collection_path).document(document_id).delete()
            logger.info(f"Document {document_id} deleted from {collection_path}")
        except Exception as e:
            logger.error(f"Error deleting document {document_id} from {collection_path}: {e}")
            raise

    def get_collection(self, collection_path: str, query_params: Optional[Dict[str, Any]] = None) -> list[Dict[str, Any]]:
        """
        Retrieves all documents from a collection, with optional query parameters.
        query_params example: {'field': 'status', 'op': '==', 'value': 'active'}
        """
        try:
            collection_ref = self.db.collection(collection_path)
            
            if query_params:
                # Basic query support (can be extended for more complex queries)
                field = query_params.get('field')
                op = query_params.get('op')
                value = query_params.get('value')
                if field and op and value is not None:
                    docs = collection_ref.where(field, op, value).stream()
                else:
                    docs = collection_ref.stream()
            else:
                docs = collection_ref.stream()

            results = []
            for doc in docs:
                results.append({"id": doc.id, **doc.to_dict()})
            logger.debug(f"Retrieved {len(results)} documents from {collection_path}")
            return results
        except Exception as e:
            logger.error(f"Error getting collection {collection_path}: {e}")
            return []

    async def get_analytics_events(
        self,
        event_type: Optional[str] = None,
        user_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves analytics events from Firestore with optional filters.
        """
        events_ref = self.db.collection("analytics_events")
        query = events_ref

        if event_type:
            query = query.where("event_type", "==", event_type)
        if user_id:
            query = query.where("user_id", "==", user_id)
        if start_date:
            query = query.where("timestamp", ">=", start_date)
        if end_date:
            query = query.where("timestamp", "<=", end_date)
        
        # Note: Firestore queries with multiple range filters or different fields
        # in range filters often require composite indexes. If you encounter errors,
        # you might need to create these indexes in your Firebase console.
        # Also, orderBy is intentionally omitted to avoid index issues as per previous instructions.

        try:
            docs = query.stream()
            events = []
            for doc in docs:
                event_data = doc.to_dict()
                events.append(event_data)
            logger.info(f"Retrieved {len(events)} analytics events.")
            return events
        except Exception as e:
            logger.error(f"Error fetching analytics events: {e}", exc_info=True)
            return []

# The instantiation of FirestoreManager will now happen in main.py,
# passing the initialized db and auth objects.
# So, remove the `firestore_manager = FirestoreManager()` line from here.

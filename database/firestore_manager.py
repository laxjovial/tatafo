# database/firestore_manager.py

import firebase_admin
from firebase_admin import credentials, firestore, auth
import logging
import json
from typing import Optional, Dict, Any

# Assuming ConfigManager is accessible
from config.config_manager import config_manager

logger = logging.getLogger(__name__)

class FirestoreManager:
    _instance = None
    _db = None
    _auth = None
    _app_id = None # Store the app_id here
    _is_initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FirestoreManager, cls).__new__(cls)
            if not cls._instance._is_initialized:
                cls._instance._initialize_firestore()
        return cls._instance

    def _initialize_firestore(self):
        if self._is_initialized:
            return

        try:
            # Try to get __app_id from Canvas globals (if running in Canvas)
            # This is primarily for frontend Firebase client SDK, but we'll try to get it
            # if it's available for consistency, though it's not strictly needed for admin SDK.
            self._app_id = config_manager.get_secret('__app_id', 'default-app-id')
            if self._app_id == 'default-app-id':
                logger.warning("Canvas-provided ____app_id not found. Falling back to default or config.")

            # For the Firebase Admin SDK, we need the service account key.
            # This should be stored in secrets.toml as firebase_admin_cert_json
            firebase_admin_cert_json_str = config_manager.get_secret('firebase_admin_cert_json')

            if not firebase_admin_cert_json_str:
                raise ValueError("Firebase Admin SDK service account key (firebase_admin_cert_json) not found in secrets.")

            # Parse the JSON string into a Python dictionary
            try:
                firebase_admin_cert = json.loads(firebase_admin_cert_json_str)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse firebase_admin_cert_json: {e}")

            # Initialize Firebase Admin SDK
            # The credentials.Certificate expects a dictionary or a path to the JSON file
            cred = credentials.Certificate(firebase_admin_cert)
            
            # Initialize the app if not already initialized
            if not firebase_admin._apps:
                firebase_admin.initialize_app(cred)
                logger.info("Firebase Admin SDK initialized successfully.")
            else:
                logger.info("Firebase Admin SDK already initialized.")

            self._db = firestore.client()
            self._auth = auth
            self._is_initialized = True
            logger.info("Firestore client and Auth initialized.")

        except ValueError as e:
            logger.error(f"Failed to initialize Firestore: {e}")
            raise RuntimeError(f"Failed to initialize Firestore: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during Firestore initialization: {e}")
            raise RuntimeError(f"An unexpected error occurred during Firestore initialization: {e}")

    @property
    def db(self):
        if not self._is_initialized:
            raise RuntimeError("Firestore not initialized. Call _initialize_firestore first.")
        return self._db

    @property
    def auth(self):
        if not self._is_initialized:
            raise RuntimeError("Firebase Auth not initialized. Call _initialize_firestore first.")
        return self._auth

    @property
    def app_id(self):
        return self._app_id

    # --- Utility methods for Firestore operations ---

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

# Instantiate the FirestoreManager as a singleton
firestore_manager = FirestoreManager()

# database/firestore_manager.py

import firebase_admin
from firebase_admin import credentials, firestore, auth
import logging
import json
from typing import Optional, Dict, Any, List, Union
import asyncio # For async operations in tests

# Import config_manager to get Firebase config from secrets.toml if __firebase_config is not available
from config.config_manager import config_manager

logger = logging.getLogger(__name__)

class FirestoreManager:
    """
    Manages all interactions with the Firestore database.
    Implemented as a singleton to ensure a single, consistent database connection.
    """
    _instance = None
    _db = None
    _app_id = None # Store the app ID for consistent pathing

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FirestoreManager, cls).__new__(cls)
            cls._instance._initialize_firestore()
        return cls._instance

    def _initialize_firestore(self):
        """
        Initializes the Firebase Admin SDK and Firestore client.
        Prioritizes Canvas-provided global variables, then secrets.toml.
        """
        if self._db is not None:
            return # Already initialized

        try:
            # Check for Canvas-provided global variables first
            firebase_config_str = None
            app_id = None

            if '____app_id' in globals(): # Note: Canvas uses `____app_id` (four underscores)
                app_id = globals()['____app_id']
                logger.info(f"Using Canvas-provided app_id: {app_id}")
            else:
                logger.warning("Canvas-provided ____app_id not found. Falling back to default or config.")
                app_id = "default-app-id" # Fallback for local testing

            if '__firebase_config' in globals(): # Note: Canvas uses `__firebase_config` (two underscores)
                firebase_config_str = globals()['__firebase_config']
                logger.info("Using Canvas-provided __firebase_config.")
            else:
                logger.warning("Canvas-provided __firebase_config not found. Falling back to secrets.toml.")
                firebase_config_str = config_manager.get_secret("firebase_config")
                if not firebase_config_str:
                    raise ValueError("Firebase config not found in Canvas globals or secrets.toml.")

            # Parse the Firebase config string
            firebase_config = json.loads(firebase_config_str)

            # Initialize Firebase Admin SDK
            if not firebase_admin._apps:
                cred = credentials.Certificate(firebase_config)
                firebase_admin.initialize_app(cred)
                logger.info("Firebase Admin SDK initialized successfully.")
            else:
                logger.info("Firebase Admin SDK already initialized.")

            self._db = firestore.client()
            self._app_id = app_id
            logger.info("Firestore client initialized.")

        except Exception as e:
            logger.error(f"Failed to initialize Firestore: {e}")
            self._db = None # Ensure db is None if initialization fails
            raise RuntimeError(f"Failed to initialize Firestore: {e}")

    @property
    def db(self):
        """Returns the Firestore client instance."""
        if self._db is None:
            # Attempt to re-initialize if it somehow became None after a failed attempt
            self._initialize_firestore()
            if self._db is None: # If still None, raise error
                raise RuntimeError("Firestore is not initialized.")
        return self._db

    @property
    def app_id(self):
        """Returns the application ID for constructing collection paths."""
        if self._app_id is None:
            # This should ideally be set during _initialize_firestore
            # but as a fallback, try to determine it.
            if '____app_id' in globals():
                self._app_id = globals()['____app_id']
            else:
                self._app_id = "default-app-id"
            logger.warning(f"app_id was not set during initialization, falling back to {self._app_id}")
        return self._app_id

    # --- Helper for Collection Paths ---
    def _get_collection_path(self, collection_name: str, user_id: Optional[str] = None, is_public: bool = False) -> str:
        """
        Constructs the Firestore collection path based on app_id, user_id, and public/private status.
        """
        if is_public:
            return f"artifacts/{self.app_id}/public/data/{collection_name}"
        elif user_id:
            return f"artifacts/{self.app_id}/users/{user_id}/{collection_name}"
        else:
            raise ValueError("user_id must be provided for private collections.")

    # --- Generic CRUD Operations ---
    async def get_document(self, collection_name: str, doc_id: str, user_id: Optional[str] = None, is_public: bool = False) -> Optional[Dict[str, Any]]:
        """Retrieves a single document."""
        try:
            collection_ref = self.db.collection(self._get_collection_path(collection_name, user_id, is_public))
            doc_ref = collection_ref.document(doc_id)
            doc = await doc_ref.get() # Use await for async get
            if doc.exists:
                return {"id": doc.id, **doc.to_dict()}
            return None
        except Exception as e:
            logger.error(f"Error getting document '{doc_id}' from '{collection_name}': {e}", exc_info=True)
            raise

    async def add_document(self, collection_name: str, data: Dict[str, Any], user_id: Optional[str] = None, is_public: bool = False) -> str:
        """Adds a new document with an auto-generated ID."""
        try:
            collection_ref = self.db.collection(self._get_collection_path(collection_name, user_id, is_public))
            doc_ref = await collection_ref.add(data) # Use await for async add
            logger.info(f"Added document to '{collection_name}' with ID: {doc_ref[1].id}")
            return doc_ref[1].id # Returns the ID of the new document
        except Exception as e:
            logger.error(f"Error adding document to '{collection_name}': {e}", exc_info=True)
            raise

    async def set_document(self, collection_name: str, doc_id: str, data: Dict[str, Any], user_id: Optional[str] = None, is_public: bool = False) -> None:
        """Sets a document by ID (creates if not exists, overwrites if exists)."""
        try:
            collection_ref = self.db.collection(self._get_collection_path(collection_name, user_id, is_public))
            doc_ref = collection_ref.document(doc_id)
            await doc_ref.set(data) # Use await for async set
            logger.info(f"Set document '{doc_id}' in '{collection_name}'.")
        except Exception as e:
            logger.error(f"Error setting document '{doc_id}' in '{collection_name}': {e}", exc_info=True)
            raise

    async def update_document(self, collection_name: str, doc_id: str, data: Dict[str, Any], user_id: Optional[str] = None, is_public: bool = False) -> None:
        """Updates an existing document (merges data)."""
        try:
            collection_ref = self.db.collection(self._get_collection_path(collection_name, user_id, is_public))
            doc_ref = collection_ref.document(doc_id)
            await doc_ref.update(data) # Use await for async update
            logger.info(f"Updated document '{doc_id}' in '{collection_name}'.")
        except Exception as e:
            logger.error(f"Error updating document '{doc_id}' in '{collection_name}': {e}", exc_info=True)
            raise

    async def delete_document(self, collection_name: str, doc_id: str, user_id: Optional[str] = None, is_public: bool = False) -> None:
        """Deletes a single document."""
        try:
            collection_ref = self.db.collection(self._get_collection_path(collection_name, user_id, is_public))
            doc_ref = collection_ref.document(doc_id)
            await doc_ref.delete() # Use await for async delete
            logger.info(f"Deleted document '{doc_id}' from '{collection_name}'.")
        except Exception as e:
            logger.error(f"Error deleting document '{doc_id}' from '{collection_name}': {e}", exc_info=True)
            raise

    async def get_collection(self, collection_name: str, user_id: Optional[str] = None, is_public: bool = False,
                             query_filters: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        Retrieves all documents from a collection, optionally with filters.
        query_filters example: [{"field": "status", "op": "==", "value": "active"}]
        """
        try:
            collection_ref = self.db.collection(self._get_collection_path(collection_name, user_id, is_public))
            
            query = collection_ref
            if query_filters:
                for f in query_filters:
                    field = f.get("field")
                    op = f.get("op")
                    value = f.get("value")
                    if field and op and value is not None:
                        query = query.where(field, op, value)

            docs = []
            # Use await for async iteration over query results
            async for doc in query.stream():
                docs.append({"id": doc.id, **doc.to_dict()})
            
            logger.info(f"Retrieved {len(docs)} documents from '{collection_name}'.")
            return docs
        except Exception as e:
            logger.error(f"Error getting collection '{collection_name}': {e}", exc_info=True)
            raise

    # --- Specific User Management Methods (for backend services to use) ---
    async def get_user_data(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves user data from the 'users' collection."""
        return await self.get_document("users", user_id, is_public=True) # Users collection is public

    async def set_user_data(self, user_id: str, data: Dict[str, Any]) -> None:
        """Sets user data in the 'users' collection."""
        await self.set_document("users", user_id, data, is_public=True)

    async def update_user_data(self, user_id: str, data: Dict[str, Any]) -> None:
        """Updates user data in the 'users' collection."""
        await self.update_document("users", user_id, data, is_public=True)

    async def get_all_user_profiles(self) -> List[Dict[str, Any]]:
        """Retrieves all user profiles from the 'users' collection."""
        return await self.get_collection("users", is_public=True)

    # --- Specific API Key Management Methods (for backend services to use) ---
    async def get_api_key(self, key_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves an API key from the 'api_keys' collection."""
        return await self.get_document("api_keys", key_id, is_public=True) # API keys can be public or admin-only

    async def add_api_key(self, data: Dict[str, Any]) -> str:
        """Adds a new API key to the 'api_keys' collection."""
        return await self.add_document("api_keys", data, is_public=True)

    async def update_api_key(self, key_id: str, data: Dict[str, Any]) -> None:
        """Updates an API key in the 'api_keys' collection."""
        await self.update_document("api_keys", key_id, data, is_public=True)

    async def delete_api_key(self, key_id: str) -> None:
        """Deletes an API key from the 'api_keys' collection."""
        await self.delete_document("api_keys", key_id, is_public=True)

    async def get_all_api_keys(self) -> List[Dict[str, Any]]:
        """Retrieves all API keys from the 'api_keys' collection."""
        return await self.get_collection("api_keys", is_public=True)

    # --- NEW: Global Configuration Management Methods (for RBAC capabilities, tiers) ---
    async def get_global_config(self, config_name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a global configuration document (e.g., 'rbac_capabilities', 'tiers')
        from the 'app_configs' public collection.
        """
        logger.info(f"Getting global config: {config_name}")
        return await self.get_document("app_configs", config_name, is_public=True)

    async def set_global_config(self, config_name: str, data: Dict[str, Any]) -> None:
        """
        Sets (creates or overwrites) a global configuration document (e.g., 'rbac_capabilities', 'tiers')
        in the 'app_configs' public collection.
        """
        logger.info(f"Setting global config: {config_name}")
        await self.set_document("app_configs", config_name, data, is_public=True)

# Instantiate the FirestoreManager as a singleton
firestore_manager = FirestoreManager()

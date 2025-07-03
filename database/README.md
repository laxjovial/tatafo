Implementing database/ Directory - Part 1: Firestore Manager
The database/ directory will house our database interaction logic. We will start by implementing firestore_manager.py to handle all interactions with our Firestore database. This is a critical step to replace the mock data we've been using in the backend API endpoints with real, persistent storage.

1. database/__init__.py
This file simply marks database as a Python package.
database/__init__.py
Jul 2, 7:32 AM

Open

2. database/firestore_manager.py
This file will contain the FirestoreManager class, responsible for all CRUD (Create, Read, Update, Delete) operations with Firestore. It will handle the Firebase Admin SDK initialization and provide methods to interact with user data, API keys, and other collections.

Important Considerations for Firestore Integration:

__app_id and __firebase_config: These are special global variables provided by the Canvas environment. We must use them for Firebase initialization. I've included logic to check for their existence and fall back to config_manager.get_secret("firebase_config") if they are not defined (e.g., when running locally or in a different environment).

Collection Paths: As per our instructions, data will be stored in specific collection paths:

Public data: /artifacts/{appId}/public/data/{your_collection_name}

Private data: /artifacts/{appId}/users/{userId}/{your_collection_name}

Error Handling: Robust try-except blocks are included for all Firestore operations.

database/firestore_manager.py
Jul 2, 7:32 AM

Open
Key Features of database/firestore_manager.py:

Singleton: Ensures only one instance of the Firestore client is created and used throughout the application.

Firebase Initialization: Handles the initialization of the Firebase Admin SDK, prioritizing Canvas global variables (____app_id, __firebase_config) and falling back to secrets.toml.

db Property: Provides easy access to the initialized Firestore client.

app_id Property: Stores and provides the application ID for consistent collection path construction.

_get_collection_path: A helper method to correctly construct Firestore collection paths based on whether data is public or user-specific, adhering to the specified security rules.

Generic CRUD Operations: get_document, add_document, set_document, update_document, delete_document, get_collection provide a flexible interface for interacting with any collection.

async/await: All Firestore operations are defined as async functions and use await for non-blocking I/O, which is crucial for FastAPI's asynchronous nature.

Specific Helper Methods: get_user_data, set_user_data, update_user_data, get_all_user_profiles, get_api_key, add_api_key, update_api_key, delete_api_key, get_all_api_keys provide convenient wrappers for common operations on users and api_keys collections.

Logging and Error Handling: Includes logging for operations and raises exceptions on errors for proper handling in calling functions.

Now that we have firestore_manager.py, we need to update the backend/api/auth_api.py and backend/api/user_api.py to use this real Firestore integration instead of the mock dictionaries.



1. database/firestore_manager.py (Updated for Global Config Management)
I'm adding get_global_config and set_global_config methods to handle documents in a dedicated app_configs collection, which will store our rbac_capabilities and tiers data.
database/firestore_manager.py (Updated for Global Config Management)
Jul 3, 2:26 AM

Open

Key Changes in database/firestore_manager.py:

New Global Config Methods:

get_global_config(config_name: str): Retrieves a document (e.g., "rbac_capabilities", "tiers") from the app_configs public collection.

set_global_config(config_name: str, data: Dict[str, Any]): Sets (creates or overwrites) a document in the app_configs public collection. This is what the admin UI/backend will use to update these configurations.

Asynchronous Operations: All new methods are async to align with Firestore's asynchronous client library.

Logging and Error Handling: Standard logging and exception handling are included.

# backend/document_upload_app.py

import streamlit as st
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import asyncio # For async operations

# Import config_manager to access Firebase configuration and other settings
from config.config_manager import config_manager
# Import analytics_tracker for logging events
from utils.analytics_tracker import log_event, initialize_analytics
# Import GCS utilities
from shared_tools.cloud_storage_utils import upload_file_to_gcs
# Import vector utilities
from shared_tools.vector_utils import load_documents_from_gcs, split_documents, create_and_store_embeddings, TEMP_UPLOAD_DIR
# Import user_manager for RBAC checks
from utils.user_manager import get_user_tier_capability

# Import Firebase Admin SDK components for backend initialization (if needed for context)
import firebase_admin
from firebase_admin import credentials, auth, firestore
import json
import os

logger = logging.getLogger(__name__)

# --- Firebase Admin SDK Initialization (for backend context) ---
if not firebase_admin._apps:
    try:
        firebase_config_str = config_manager.get_secret("firebase_config")
        if not firebase_config_str:
            raise ValueError("Firebase configuration not found in secrets.")
        
        firebase_config = json.loads(firebase_config_str)
        
        if os.environ.get("FIREBASE_ADMIN_CERT"):
            cred = credentials.Certificate(json.loads(os.environ.get("FIREBASE_ADMIN_CERT")))
        else:
            logger.warning("FIREBASE_ADMIN_CERT environment variable not found. Firebase Admin SDK functionality may be limited.")
            cred = credentials.Certificate({
                "type": "service_account",
                "project_id": firebase_config.get("projectId", "mock-project-id"),
                "private_key_id": "mock-key-id",
                "private_key": "-----BEGIN PRIVATE KEY-----\nMOCK_PRIVATE_KEY\n-----END PRIVATE KEY-----\n",
                "client_email": "mock-client@mock-project-id.iam.gserviceaccount.com",
                "client_id": "mock-client-id",
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "client_x509_cert_url": f"https://www.googleapis.com/robot/v1/metadata/x509/mock-client%40{firebase_config.get('projectId', 'mock-project-id')}.iam.gserviceaccount.com",
                "universe_domain": "googleapis.com"
            })

        firebase_admin.initialize_app(cred)
        logger.info("Firebase Admin SDK initialized successfully in document_upload_app.")
    except Exception as e:
        logger.error(f"Error initializing Firebase Admin SDK in document_upload_app: {e}")

# Initialize analytics_tracker for backend context
if 'analytics_initialized_backend' not in globals(): # Use globals() for module-level check
    if firebase_admin._apps:
        try:
            db_instance = firestore.client()
            auth_instance = auth
            app_id_for_analytics = config_manager.get("app_id", firebase_config.get("projectId", "default-streamlit-app-id"))
            initialize_analytics(db_instance, auth_instance, app_id_for_analytics, "backend_system_user")
            globals()['analytics_initialized_backend'] = True
            logger.info("Analytics tracker initialized for document_upload_app with live Firebase.")
        except Exception as e:
            logger.error(f"Failed to initialize analytics with live Firebase Admin SDK in document_upload_app: {e}")
            mock_db = type('FirestoreMock', (object,), {'collection': lambda s, path: type('CollectionMock', (object,), {'add': lambda s, data: None})()})()
            mock_auth = type('AuthMock', (object,), {'currentUser': type('CurrentUserMock', (object,), {'uid': None})()})()
            app_id_for_analytics = config_manager.get("app_id", "default-streamlit-app-id")
            initialize_analytics(mock_db, mock_auth, app_id_for_analytics, "backend_system_user")
            globals()['analytics_initialized_backend'] = True
            logger.warning("Analytics tracker initialized with mock Firebase for document_upload_app.")
    else:
        mock_db = type('FirestoreMock', (object,), {'collection': lambda s, path: type('CollectionMock', (object,), {'add': lambda s, data: None})()})()
        mock_auth = type('AuthMock', (object,), {'currentUser': type('CurrentUserMock', (object,), {'uid': None})()})()
        app_id_for_analytics = config_manager.get("app_id", "default-streamlit-app-id")
        initialize_analytics(mock_db, mock_auth, app_id_for_analytics, "backend_system_user")
        globals()['analytics_initialized_backend'] = True
        logger.warning("Analytics tracker initialized with mock Firebase for document_upload_app (Admin SDK not available).")


def app():
    st.title("⬆️ Upload & Index Documents")
    st.info("Upload documents (PDF, DOCX, TXT) to build your custom knowledge base for various domains.")

    # Ensure user is logged in
    if "logged_in" not in st.session_state or not st.session_state.logged_in:
        st.warning("Please log in to upload documents.")
        asyncio.run(log_event('page_view', {
            'page_name': 'DocumentUpload',
            'status': 'access_denied',
            'reason': 'not_logged_in'
        }, user_id='unauthenticated', success=False))
        return

    user_id = st.session_state.get('user_id_from_backend')
    user_token = st.session_state.get('user_token') # For RBAC checks

    if not user_id or not user_token:
        st.error("User authentication information missing. Please log in again.")
        asyncio.run(log_event('page_view', {
            'page_name': 'DocumentUpload',
            'status': 'access_denied',
            'reason': 'missing_auth_info'
        }, user_id='unknown_user', success=False))
        return

    # RBAC check for document upload capability
    if not get_user_tier_capability(user_token, 'document_upload_enabled', False):
        st.warning("Your current subscription tier does not allow document uploads. Please upgrade your plan.")
        asyncio.run(log_event('page_view', {
            'page_name': 'DocumentUpload',
            'status': 'access_denied',
            'reason': 'rbac_denied'
        }, user_id=user_id, success=False))
        return

    # Log successful page view
    asyncio.run(log_event('page_view', {
        'page_name': 'DocumentUpload',
        'status': 'accessed',
        'user_id': user_id
    }, user_id=user_id, success=True))

    # Get configurable RAG parameters
    chunk_size = config_manager.get('rag.chunk_size', 500)
    chunk_overlap = config_manager.get('rag.chunk_overlap', 50)
    embedding_provider = config_manager.get('rag.embedding_provider', 'openai')
    embedding_model_name = config_manager.get('rag.embedding_model_name', 'text-embedding-ada-002')

    st.subheader("Document Details")
    
    # Allow user to select a domain/collection name
    available_domains = config_manager.get('rag.available_domains', ['general', 'medical', 'legal', 'finance', 'education', 'sports', 'news'])
    selected_domain = st.selectbox(
        "Select Document Domain (Collection Name)",
        options=available_domains,
        key="doc_domain_select",
        help="Documents will be indexed into this domain's knowledge base."
    )

    uploaded_file = st.file_uploader(
        "Choose a document (PDF, DOCX, TXT)",
        type=["pdf", "docx", "txt"],
        key="document_uploader"
    )

    if uploaded_file is not None:
        file_name = uploaded_file.name
        file_size = uploaded_file.size
        file_type = uploaded_file.type

        st.write(f"**File Name:** {file_name}")
        st.write(f"**File Type:** {file_type}")
        st.write(f"**File Size:** {file_size / (1024 * 1024):.2f} MB")

        if st.button("Upload & Index"):
            with st.spinner("Uploading and processing document..."):
                try:
                    # 1. Save uploaded file to a temporary local path
                    temp_local_file_path = TEMP_UPLOAD_DIR / user_id / file_name
                    temp_local_file_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(temp_local_file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    logger.info(f"Saved uploaded file temporarily to {temp_local_file_path}")
                    
                    # 2. Upload the file to GCS
                    # GCS blob path structure: user_uploads/<user_id>/<domain>/<file_name>
                    gcs_blob_name = f"user_uploads/{user_id}/{selected_domain}/{file_name}"
                    gcs_uri = await upload_file_to_gcs(str(temp_local_file_path), gcs_blob_name, user_id=user_id)

                    if not gcs_uri:
                        raise Exception("Failed to upload file to Google Cloud Storage.")

                    st.success(f"File '{file_name}' uploaded to GCS: {gcs_uri}")
                    
                    # 3. Load document from GCS for processing (it will be downloaded temporarily again by vector_utils)
                    # We pass the same GCS blob path, vector_utils handles the temp download/cleanup
                    documents = await load_documents_from_gcs(gcs_blob_name, user_id=user_id)
                    if not documents:
                        raise Exception("Failed to load document content for indexing.")

                    # 4. Split documents into chunks
                    chunks = split_documents(documents, chunk_size, chunk_overlap)
                    st.info(f"Document split into {len(chunks)} chunks.")

                    # 5. Create embeddings and store in ChromaDB
                    # The collection name will be user_id/selected_domain
                    collection_name = f"{user_id}_{selected_domain}" # Unique collection per user+domain
                    await create_and_store_embeddings(
                        chunks,
                        collection_name,
                        user_id=user_id,
                        embedding_provider=embedding_provider,
                        embedding_model_name=embedding_model_name
                    )
                    st.success(f"Document indexed successfully into '{selected_domain}' knowledge base!")
                    
                    asyncio.run(log_event('document_upload_and_index', {
                        'file_name': file_name,
                        'file_size': file_size,
                        'domain': selected_domain,
                        'gcs_uri': gcs_uri,
                        'num_chunks': len(chunks),
                        'status': 'success'
                    }, user_id=user_id, success=True))

                except Exception as e:
                    st.error(f"Error processing document: {e}")
                    logger.error(f"Document upload and indexing failed for user {user_id}, file {file_name}: {e}", exc_info=True)
                    asyncio.run(log_event('document_upload_and_index', {
                        'file_name': file_name,
                        'file_size': file_size,
                        'domain': selected_domain,
                        'status': 'failure',
                        'error_message': str(e)
                    }, user_id=user_id, success=False, error_message=str(e)))
                finally:
                    # Ensure the temporary local file is removed after processing
                    if temp_local_file_path.exists():
                        os.remove(temp_local_file_path)
                        logger.info(f"Cleaned up temporary local file: {temp_local_file_path}")

    st.markdown("---")
    st.markdown("Once indexed, you can query your documents using the relevant AI Assistant or Query Tools.")


# Example of how to run this app standalone for testing
if __name__ == "__main__":
    # Mock session state for standalone testing
    if "user_id_from_backend" not in st.session_state:
        st.session_state.user_id_from_backend = "mock_premium_uid_upload"
        st.session_state.user_token = "mock_token_premium_upload"
        st.session_state.logged_in = True
    
    # Mock RBAC capability for testing
    import unittest.mock as mock
    original_get_user_tier_capability = get_user_tier_capability
    
    def mock_get_user_tier_capability(user_token, capability_key, default_value):
        if capability_key == 'document_upload_enabled':
            return True # Enable for testing
        return original_get_user_tier_capability(user_token, capability_key, default_value)
    
    # Patch the actual function
    import sys
    sys.modules['utils.user_manager'].get_user_tier_capability = mock_get_user_tier_capability

    # Mock GCS and vector_utils functions
    mock_upload_file_to_gcs = AsyncMock(return_value="gs://mock-bucket/user_uploads/mock_premium_uid_upload/general/test_doc.pdf")
    mock_load_documents_from_gcs = AsyncMock(return_value=[
        MagicMock(page_content="Mock doc content 1.", metadata={"source": "test_doc.pdf", "page": 0}),
        MagicMock(page_content="Mock doc content 2.", metadata={"source": "test_doc.pdf", "page": 1})
    ])
    mock_split_documents = MagicMock(return_value=[
        MagicMock(page_content="Chunk 1.", metadata={}),
        MagicMock(page_content="Chunk 2.", metadata={})
    ])
    mock_create_and_store_embeddings = AsyncMock(return_value=MagicMock())

    with patch('shared_tools.cloud_storage_utils.upload_file_to_gcs', new=mock_upload_file_to_gcs), \
         patch('shared_tools.vector_utils.load_documents_from_gcs', new=mock_load_documents_from_gcs), \
         patch('shared_tools.vector_utils.split_documents', new=mock_split_documents), \
         patch('shared_tools.vector_utils.create_and_store_embeddings', new=mock_create_and_store_embeddings):
        
        # Initialize analytics for the test run if not already done
        if 'analytics_initialized_backend' not in st.session_state:
            mock_db_for_analytics = type('FirestoreMock', (object,), {'collection': lambda s, path: type('CollectionMock', (object,), {'add': lambda s, data: asyncio.sleep(0.01)})()})()
            mock_auth_for_analytics = type('AuthMock', (object,), {'currentUser': type('CurrentUserMock', (object,), {'uid': "test_cli_user"})()})()
            initialize_analytics(mock_db_for_analytics, mock_auth_for_analytics, "test-app-id-cli", "test_cli_user")
            st.session_state.analytics_initialized_backend = True

        st.write("Running standalone Document Upload App test. Upload a dummy file to test.")
        app()

        # Restore original functions after testing
        sys.modules['utils.user_manager'].get_user_tier_capability = original_get_user_tier_capability

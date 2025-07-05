# backend/ai_assistant_app.py

import streamlit as st
import logging
import asyncio
import httpx
import json
import os
from typing import List, Dict, Any, Optional

# Import config_manager to access configurations
from config.config_manager import config_manager
# Import analytics_tracker for logging events
from utils.analytics_tracker import log_event, initialize_analytics
# Import UserManager for backend interactions
from utils.user_manager import UserManager
# Import DocumentQueryManager for document querying capabilities
from utils.document_query_manager import DocumentQueryManager

# Import Firebase Admin SDK components for backend initialization (if needed for context)
import firebase_admin
from firebase_admin import credentials, auth, firestore

logger = logging.getLogger(__name__)

# --- Firebase Admin SDK Initialization (for backend context) ---
# This block ensures Firebase Admin SDK is initialized once per session/module load
# It's crucial for analytics and potentially other backend operations if this script
# were to directly interact with Firebase Admin SDK for user management or data.
if not firebase_admin._apps:
    try:
        firebase_config_str = config_manager.get_secret("firebase_config")
        if not firebase_config_str:
            raise ValueError("Firebase configuration not found in secrets.")
        
        firebase_config = json.loads(firebase_config_str)
        
        # Prioritize environment variable for service account key in production
        if os.environ.get("FIREBASE_ADMIN_CERT"):
            cred = credentials.Certificate(json.loads(os.environ.get("FIREBASE_ADMIN_CERT")))
        else:
            logger.warning("FIREBASE_ADMIN_CERT environment variable not found. Using mock credentials for Firebase Admin SDK initialization. This is not suitable for production.")
            # Fallback to mock credentials for local development if env var is not set
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
        logger.info("Firebase Admin SDK initialized successfully in ai_assistant_app.")
    except Exception as e:
        logger.error(f"Error initializing Firebase Admin SDK in ai_assistant_app: {e}")

# Initialize analytics_tracker for backend context
# This ensures that analytics logging can occur even from within Streamlit apps
if 'analytics_initialized_ai_assistant' not in globals():
    if firebase_admin._apps:
        try:
            db_instance = firestore.client()
            auth_instance = auth
            app_id_for_analytics = config_manager.get("app_id", firebase_config.get("projectId", "default-streamlit-app-id"))
            initialize_analytics(db_instance, auth_instance, app_id_for_analytics, "backend_system_user")
            globals()['analytics_initialized_ai_assistant'] = True
            logger.info("Analytics tracker initialized for ai_assistant_app with live Firebase.")
        except Exception as e:
            logger.error(f"Failed to initialize analytics with live Firebase Admin SDK in ai_assistant_app: {e}")
            # Fallback to mock if live Firebase fails
            mock_db = type('FirestoreMock', (object,), {'collection': lambda s, path: type('CollectionMock', (object,), {'add': lambda s, data: None})()})()
            mock_auth = type('AuthMock', (object,), {'currentUser': type('CurrentUserMock', (object,), {'uid': None})()})()
            app_id_for_analytics = config_manager.get("app_id", "default-streamlit-app-id")
            initialize_analytics(mock_db, mock_auth, app_id_for_analytics, "backend_system_user")
            globals()['analytics_initialized_ai_assistant'] = True
            logger.warning("Analytics tracker initialized with mock Firebase for ai_assistant_app.")
    else:
        # If Firebase Admin SDK is not initialized at all
        mock_db = type('FirestoreMock', (object,), {'collection': lambda s, path: type('CollectionMock', (object,), {'add': lambda s, data: None})()})()
        mock_auth = type('AuthMock', (object,), {'currentUser': type('CurrentUserMock', (object,), {'uid': None})()})()
        app_id_for_analytics = config_manager.get("app_id", "default-streamlit-app-id")
        initialize_analytics(mock_db, mock_auth, app_id_for_analytics, "backend_system_user")
        globals()['analytics_initialized_ai_assistant'] = True
        logger.warning("Analytics tracker initialized with mock Firebase for ai_assistant_app (Admin SDK not available).")


# Initialize UserManager and DocumentQueryManager
user_manager = UserManager()
document_query_manager = DocumentQueryManager()

# Define the base URL for the FastAPI backend
FASTAPI_BACKEND_URL = config_manager.get("fastapi_backend_url", "http://localhost:8000")

async def call_llm_api(prompt: str, user_id: str, id_token: str) -> str:
    """
    Calls the LLM API endpoint in the FastAPI backend.
    Includes user_id and id_token for context and authentication.
    """
    url = f"{FASTAPI_BACKEND_URL}/llm/generate"
    headers = {
        "Authorization": f"Bearer {id_token}",
        "Content-Type": "application/json"
    }
    payload = {
        "prompt": prompt,
        "user_id": user_id # Pass user_id for backend logging/context
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=payload, timeout=60.0) # Increased timeout
            response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
            result = response.json()
            if result.get("success"):
                asyncio.create_task(log_event(
                    'llm_interaction',
                    {'prompt': prompt, 'response_length': len(result.get('response', '')), 'status': 'success'},
                    user_id=user_id, success=True
                ))
                return result.get("response", "No response from LLM.")
            else:
                error_message = result.get("message", "Unknown error from LLM API.")
                asyncio.create_task(log_event(
                    'llm_interaction',
                    {'prompt': prompt, 'status': 'failure', 'error': error_message},
                    user_id=user_id, success=False, error_message=error_message
                ))
                return f"Error from LLM API: {error_message}"
    except httpx.RequestError as e:
        error_message = f"Network or HTTP error calling LLM API: {e}"
        asyncio.create_task(log_event(
            'llm_interaction',
            {'prompt': prompt, 'status': 'failure', 'error': error_message},
            user_id=user_id, success=False, error_message=error_message
        ))
        return f"Could not connect to LLM service: {e}"
    except httpx.HTTPStatusError as e:
        error_message = f"HTTP error calling LLM API: {e.response.status_code} - {e.response.text}"
        asyncio.create_task(log_event(
            'llm_interaction',
            {'prompt': prompt, 'status': 'failure', 'error': error_message},
            user_id=user_id, success=False, error_message=error_message
        ))
        return f"Error from LLM service: {e.response.status_code} - {e.response.text}"
    except json.JSONDecodeError:
        error_message = "Failed to parse JSON response from LLM API."
        asyncio.create_task(log_event(
            'llm_interaction',
            {'prompt': prompt, 'status': 'failure', 'error': error_message},
            user_id=user_id, success=False, error_message=error_message
        ))
        return f"Invalid JSON response from LLM API."
    except Exception as e:
        error_message = f"An unexpected error occurred calling LLM API: {e}"
        asyncio.create_task(log_event(
            'llm_interaction',
            {'prompt': prompt, 'status': 'failure', 'error': error_message},
            user_id=user_id, success=False, error_message=error_message
        ))
        return f"An unexpected error occurred: {e}"

async def query_documents_async(query_text: str, user_id: str, id_token: str, k: int) -> List[Dict[str, Any]]:
    """
    Asynchronously queries documents using the DocumentQueryManager.
    """
    response = await document_query_manager.query_documents(query_text, user_id, id_token, k)
    if response.get("success"):
        asyncio.create_task(log_event(
            'document_query',
            {'query': query_text, 'results_count': len(response.get('results', [])), 'status': 'success'},
            user_id=user_id, success=True
        ))
        return response.get("results", [])
    else:
        error_message = response.get("message", "Failed to query documents.")
        asyncio.create_task(log_event(
            'document_query',
            {'query': query_text, 'status': 'failure', 'error': error_message},
            user_id=user_id, success=False, error_message=error_message
        ))
        st.error(f"Error querying documents: {error_message}")
        return []

def app():
    st.title("🤖 AI Assistant")
    st.info("Interact with the AI, ask questions, and query your uploaded documents.")

    # Ensure user is logged in
    if not user_manager.st.session_state.is_authenticated:
        st.warning("Please log in to use the AI Assistant.")
        asyncio.run(log_event('page_view', {
            'page_name': 'AIAssistant',
            'status': 'access_denied',
            'reason': 'not_logged_in'
        }, user_id='unauthenticated', success=False))
        return

    user_id = user_manager.st.session_state.user_id
    id_token = user_manager.st.session_state.id_token
    user_capabilities = user_manager.st.session_state.user_capabilities

    # Log successful page view
    asyncio.run(log_event('page_view', {
        'page_name': 'AIAssistant',
        'status': 'accessed',
        'user_id': user_id
    }, user_id=user_id, success=True))

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input for user query
    user_query = st.chat_input("Ask the AI something or query your documents...")

    if user_query:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Determine if document querying is enabled for the user's tier
        document_query_enabled = user_capabilities.get('document_query_enabled', False)
        document_query_max_results_k = user_capabilities.get('document_query_max_results_k', 3) # Default k

        ai_response_placeholder = st.empty() # Placeholder for AI response
        with ai_response_placeholder.container():
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    llm_response = ""
                    document_results = []
                    
                    # Heuristic: If the query seems like a document query, try that first
                    # This is a simple heuristic; a more advanced agent would decide based on tools
                    is_document_query = False
                    if document_query_enabled:
                        doc_query_keywords = ["document", "file", "report", "data", "uploaded", "my files", "what about"]
                        if any(keyword in user_query.lower() for keyword in doc_query_keywords):
                            is_document_query = True
                            st.info("Attempting to query documents...")
                            document_results = asyncio.run(query_documents_async(user_query, user_id, id_token, document_query_max_results_k))
                            
                            if document_results:
                                doc_context = "\n\n--- Relevant Documents ---\n"
                                for i, doc in enumerate(document_results):
                                    doc_context += f"Document {i+1} (Source: {doc.get('source', 'N/A')}):\n"
                                    doc_context += f"{doc.get('content', 'No content available.')}\n\n"
                                doc_context += "--------------------------\n\n"
                                
                                # Prepend document context to the prompt for the LLM
                                prompt_with_context = f"Based on the following relevant information:\n{doc_context}\n\nAnswer the user's question: {user_query}"
                                llm_response = asyncio.run(call_llm_api(prompt_with_context, user_id, id_token))
                            else:
                                llm_response = "I couldn't find any relevant information in your documents for that query. Perhaps I can answer from my general knowledge base?"
                                # Fallback to general LLM if no document results
                                llm_response += "\n\n" + asyncio.run(call_llm_api(user_query, user_id, id_token))
                        
                    if not is_document_query or not document_query_enabled:
                        # If not a document query, or document querying is not enabled, call general LLM
                        llm_response = asyncio.run(call_llm_api(user_query, user_id, id_token))

                    st.markdown(llm_response)
                    st.session_state.messages.append({"role": "assistant", "content": llm_response})

                    if document_results:
                        st.subheader("Document Sources:")
                        for doc in document_results:
                            st.markdown(f"- **{doc.get('source', 'Unknown Source')}**: {doc.get('summary', 'No summary available.')}")
                            if doc.get('url'):
                                st.markdown(f"  [View Original]({doc['url']})")

# Example of how to run this app standalone for testing
if __name__ == "__main__":
    # Mock session state for standalone testing
    if "user_id" not in st.session_state:
        st.session_state.user_id = "mock_user_uid_ai"
        st.session_state.username = "MockUserAI"
        st.session_state.email = "mockai@example.com"
        st.session_state.id_token = "mock_ai_token_for_backend"
        st.session_state.is_authenticated = True
        st.session_state.user_profile = {"tier": "pro", "roles": ["user"]}
        st.session_state.user_capabilities = {
            'analytics_access': False,
            'document_upload_enabled': True,
            'document_query_enabled': True,
            'document_query_max_results_k': 5, # Set for testing document query
            'llm_access_enabled': True,
            # ... other capabilities
        }
        st.session_state.messages = [] # Initialize messages for testing

    # Mock UserManager and DocumentQueryManager methods
    import unittest.mock as mock
    from unittest.mock import patch

    mock_user_manager_instance = mock.MagicMock(spec=UserManager)
    mock_user_manager_instance.st = st # Allow mock to access st.session_state
    
    mock_document_query_manager_instance = mock.MagicMock(spec=DocumentQueryManager)
    mock_document_query_manager_instance.query_documents = mock.AsyncMock(
        side_effect=[
            {"success": True, "results": [
                {"source": "Report A", "content": "The Q3 earnings were higher than expected due to strong sales in Europe.", "summary": "Q3 earnings report."},
                {"source": "Memo B", "content": "Our new marketing strategy focuses on digital channels and influencer partnerships.", "summary": "Marketing strategy overview."}
            ]},
            {"success": True, "results": []} # For subsequent calls
        ]
    )

    # Mock the httpx.AsyncClient for LLM API calls
    async def mock_post_llm(*args, **kwargs):
        mock_response = mock.MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        
        prompt = kwargs['json']['prompt']
        if "Q3 earnings" in prompt:
            mock_response.json.return_value = {"success": True, "response": "Based on the documents, the Q3 earnings were indeed higher than expected, driven by strong sales in Europe. The new marketing strategy focuses on digital channels."}
        elif "general knowledge" in prompt:
            mock_response.json.return_value = {"success": True, "response": "This is a general knowledge response."}
        else:
            mock_response.json.return_value = {"success": True, "response": f"AI response to: '{prompt}'"}
        return mock_response

    mock_httpx_client = mock.AsyncMock()
    mock_httpx_client.post.side_effect = mock_post_llm
    
    # Patch httpx.AsyncClient to return our mock client
    with patch('httpx.AsyncClient', return_value=mock_httpx_client):
        with patch('utils.user_manager.UserManager', return_value=mock_user_manager_instance):
            with patch('utils.document_query_manager.DocumentQueryManager', return_value=mock_document_query_manager_instance):
                # Initialize analytics for the test run if not already done
                if 'analytics_initialized_ai_assistant' not in st.session_state:
                    mock_db_for_analytics = type('FirestoreMock', (object,), {'collection': lambda s, path: type('CollectionMock', (object,), {'add': lambda s, data: asyncio.sleep(0.01)})()})()
                    mock_auth_for_analytics = type('AuthMock', (object,), {'currentUser': type('CurrentUserMock', (object,), {'uid': "test_cli_user_ai"})()})()
                    initialize_analytics(mock_db_for_analytics, mock_auth_for_analytics, "test-app-id-cli-ai", "test_cli_user_ai")
                    st.session_state.analytics_initialized_ai_assistant = True

                st.write("Running standalone AI Assistant App test. Try asking 'What about Q3 earnings in my documents?' or a general question.")
                app()

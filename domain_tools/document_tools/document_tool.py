# domain_tools/document_tools/document_tool.py

import logging
from typing import Optional, Dict, Any, List
import asyncio

# Import the actual vector utility functions from shared_tools.vector_utils
# These functions will now receive the necessary managers as arguments.
from shared_tools.vector_utils import query_documents as vector_query_documents
from shared_tools.vector_utils import process_uploaded_document as vector_process_uploaded_document

# Import config_manager to access configurations
from config.config_manager import config_manager
# Import user_manager for RBAC checks
from utils.user_manager import get_user_tier_capability
# Import analytics_tracker
from utils import analytics_tracker # Import the module

logger = logging.getLogger(__name__)

# This module will contain the actual tool functions for document operations.
# These functions will be methods of the DocumentTools class.

async def query_uploaded_docs_internal(
    query_text: str,
    user_id: str,
    firestore_manager: Any,
    cloud_storage_utils: Any,
    config_manager_instance: Any,
    log_event_func: Any,
    collection_name: Optional[str] = None,
    k: int = 5
) -> str:
    """
    Internal function to query uploaded documents for relevant information.
    This function will be called by the DocumentTools class.

    Args:
        query_text (str): The query string to search for.
        user_id (str): The ID of the user performing the query.
        firestore_manager (Any): The FirestoreManager instance.
        cloud_storage_utils (Any): The CloudStorageUtilsWrapper instance.
        config_manager_instance (Any): The ConfigManager instance.
        log_event_func (Any): The analytics logging function.
        collection_name (str, optional): The specific collection to query. Defaults to None (user's default).
        k (int, optional): The number of top results to return. Defaults to 5.

    Returns:
        str: A formatted string of the most relevant document chunks, or a message indicating no results.
    """
    logger.info(f"Internal: query_uploaded_docs_internal called for user {user_id} with query: '{query_text}' (k={k})")

    # RBAC check is handled at the API endpoint level, but can be re-checked here if needed.
    # For now, assuming the FastAPI endpoint already verified access.

    try:
        # Call the core vector_query_documents function, passing all necessary dependencies
        results = await vector_query_documents(
            query_text=query_text,
            user_id=user_id,
            collection_name=collection_name,
            k=k,
            firestore_manager=firestore_manager, # Pass the instance
            cloud_storage_utils=cloud_storage_utils, # Pass the instance
            config_manager=config_manager_instance # Pass the instance
        )

        if results:
            response_str = f"Found {len(results)} relevant document chunks for your query:\n\n"
            for i, result in enumerate(results):
                source = result.get('source', 'N/A')
                page_content = result.get('page_content', 'No content available')
                score = result.get('score', 'N/A')
                response_str += f"--- Chunk {i+1} (Score: {score:.4f}) ---\n"
                response_str += f"Source: {source}\n"
                response_str += f"Content: {page_content}\n\n"
            
            # Log successful tool usage
            await log_event_func(
                'tool_usage',
                {'tool': 'document_tools.query_uploaded_docs', 'query': query_text, 'num_results': len(results), 'status': 'success'},
                user_id=user_id,
                success=True
            )
            return response_str
        else:
            message = f"No relevant information found in your uploaded documents for the query: '{query_text}'."
            # Log no results as a successful tool execution but with no relevant data
            await log_event_func(
                'tool_usage',
                {'tool': 'document_tools.query_uploaded_docs', 'query': query_text, 'num_results': 0, 'status': 'no_results'},
                user_id=user_id,
                success=True,
                error_message="No relevant results found"
            )
            return message
    except Exception as e:
        error_msg = f"Error querying documents for user {user_id}: {e}"
        logger.error(error_msg, exc_info=True)
        # Log failed tool usage
        await log_event_func(
            'tool_usage',
            {'tool': 'document_tools.query_uploaded_docs', 'query': query_text, 'status': 'failure', 'error': str(e)},
            user_id=user_id,
            success=False,
            error_message=error_msg
        )
        return f"An error occurred while querying your documents: {e}"


# --- Test Functions (for direct execution of this file) ---
async def run_document_tests():
    """Runs a series of tests for the document tools."""
    print("--- Running Document Tool Tests ---")
    test_user_pro = "test_user_pro_doc_123" # A dummy user token for testing

    # Mock dependencies for testing
    class MockFirestoreManager:
        def __init__(self):
            self.mock_data = {} # Simulate Firestore documents for vector store metadata
            self.analytics_events = []

        async def get_document(self, path):
            # Simulate fetching vector store metadata
            if "vector_stores" in path:
                # Mock a vector store entry
                return {
                    "user_id": test_user_pro,
                    "collection_name": "default_collection",
                    "vector_store_path": "mock_vector_store_path/test_user_pro/default_collection.faiss",
                    "doc_metadata": {
                        "doc1.txt": {"source": "doc1.txt", "chunks": 2},
                        "doc2.pdf": {"source": "doc2.pdf", "chunks": 3}
                    }
                }
            return None

        async def add_document(self, collection_path, data):
            print(f"MockFirestoreManager: Added document to {collection_path}: {data}")
            if "analytics_events" in collection_path:
                self.analytics_events.append(data)
            return {"id": "mock_doc_id"}

        # Add other methods if needed by vector_utils functions
        async def update_document(self, doc_ref, data):
            print(f"MockFirestoreManager: Updated document {doc_ref}: {data}")
            return {"success": True}

    class MockCloudStorageUtils:
        def __init__(self):
            self.mock_files = {} # Simulate GCS files

        async def download_file_from_gcs(self, blob_name, destination_path):
            # Simulate downloading a vector store file
            if "mock_vector_store_path" in blob_name and ".faiss" in blob_name:
                # Create a dummy FAISS index file
                from faiss import IndexFlatL2
                import numpy as np
                d = 128 # dimension
                nb = 100 # database size
                np.random.seed(1234) # make reproducible
                xb = np.random.random((nb, d)).astype('float32')
                xb[:, 0] += np.arange(nb) / 1000.
                index = IndexFlatL2(d)
                index.add(xb)
                
                # Ensure the directory exists
                Path(destination_path).parent.mkdir(parents=True, exist_ok=True)
                index_path = Path(destination_path)
                from faiss import write_index
                write_index(index, str(index_path))
                print(f"MockCloudStorageUtils: Simulated download of {blob_name} to {destination_path}")
                return True
            return False

        async def upload_file_to_gcs(self, *args, **kwargs):
            print(f"MockCloudStorageUtils: Simulated upload of {kwargs.get('blob_name')}")
            return True

        async def delete_file_from_gcs(self, *args, **kwargs):
            print(f"MockCloudStorageUtils: Simulated delete of {kwargs.get('blob_name')}")
            return True

    class MockConfigManager:
        def get(self, key, default=None):
            if key == "vector_db.embedding_model":
                return "mock_embedding_model"
            if key == "vector_db.index_dimension":
                return 128
            if key == "analytics.log_tool_usage":
                return True
            return default

    class MockEmbeddingModel:
        def embed_query(self, text):
            # Mock embedding for testing
            return [float(i) for i in range(128)] # Return a dummy embedding

        def embed_documents(self, texts):
            return [[float(i) for i in range(128)] for _ in texts]


    mock_firestore_manager = MockFirestoreManager()
    mock_cloud_storage_utils = MockCloudStorageUtils()
    mock_config_manager = MockConfigManager()
    mock_log_event = analytics_tracker.log_event # Use the real log_event which uses the mock db

    # Temporarily override get_embedding_model in vector_utils for testing
    original_get_embedding_model = vector_query_documents.__globals__.get('get_embedding_model')
    vector_query_documents.__globals__['get_embedding_model'] = lambda: MockEmbeddingModel()


    try:
        # Test 1: query_uploaded_docs_internal (successful query)
        print("\n--- Test 1: query_uploaded_docs_internal (Successful Query) ---")
        query = "What is the main topic of document 1?"
        result = await query_uploaded_docs_internal(
            query_text=query,
            user_id=test_user_pro,
            firestore_manager=mock_firestore_manager,
            cloud_storage_utils=mock_cloud_storage_utils,
            config_manager_instance=mock_config_manager,
            log_event_func=mock_log_event,
            collection_name="default_collection",
            k=2
        )
        print(f"Result: {result}")
        assert "Found 2 relevant document chunks" in result
        assert any(e['tool_name'] == 'document_tools.query_uploaded_docs' and e['success'] for e in mock_firestore_manager.analytics_events)
        print("Test 1 Passed.")
        mock_firestore_manager.analytics_events.clear() # Clear for next test

        # Test 2: query_uploaded_docs_internal (no results)
        print("\n--- Test 2: query_uploaded_docs_internal (No Results) ---")
        query = "This query will yield no results."
        # Temporarily mock vector_query_documents to return empty list
        original_vector_query_documents = vector_query_documents
        vector_query_documents.__globals__['vector_query_documents'] = lambda *args, **kwargs: asyncio.sleep(0.01) or [] # Mock empty results

        result = await query_uploaded_docs_internal(
            query_text=query,
            user_id=test_user_pro,
            firestore_manager=mock_firestore_manager,
            cloud_storage_utils=mock_cloud_storage_utils,
            config_manager_instance=mock_config_manager,
            log_event_func=mock_log_event,
            collection_name="default_collection",
            k=2
        )
        print(f"Result: {result}")
        assert "No relevant information found" in result
        assert any(e['tool_name'] == 'document_tools.query_uploaded_docs' and e['status'] == 'no_results' for e in mock_firestore_manager.analytics_events)
        print("Test 2 Passed.")
        mock_firestore_manager.analytics_events.clear()
        vector_query_documents.__globals__['vector_query_documents'] = original_vector_query_documents # Restore

        # Test 3: query_uploaded_docs_internal (error scenario)
        print("\n--- Test 3: query_uploaded_docs_internal (Error Scenario) ---")
        query = "trigger error"
        # Temporarily mock vector_query_documents to raise an exception
        original_vector_query_documents = vector_query_documents
        def mock_error_query_documents(*args, **kwargs):
            raise ValueError("Simulated query error")
        vector_query_documents.__globals__['vector_query_documents'] = mock_error_query_documents

        result = await query_uploaded_docs_internal(
            query_text=query,
            user_id=test_user_pro,
            firestore_manager=mock_firestore_manager,
            cloud_storage_utils=mock_cloud_storage_utils,
            config_manager_instance=mock_config_manager,
            log_event_func=mock_log_event,
            collection_name="default_collection",
            k=1
        )
        print(f"Result: {result}")
        assert "An error occurred while querying your documents" in result
        assert any(e['tool_name'] == 'document_tools.query_uploaded_docs' and not e['success'] for e in mock_firestore_manager.analytics_events)
        print("Test 3 Passed.")
        mock_firestore_manager.analytics_events.clear()
        vector_query_documents.__globals__['vector_query_documents'] = original_vector_query_documents # Restore

        print("\nAll document_tool tests completed.")

    finally:
        # Restore original get_embedding_model
        if original_get_embedding_model:
            vector_query_documents.__globals__['get_embedding_model'] = original_get_embedding_model
        else:
            del vector_query_documents.__globals__['get_embedding_model'] # If it wasn't there originally

# Ensure tests are only run when the script is executed directly
if __name__ == "__main__":
    asyncio.run(run_document_tests())

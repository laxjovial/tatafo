# shared_tools/vector_utils.py

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import os
import shutil
import base64
import asyncio

# Import necessary components for vector storage and processing
# Assuming you have a vector database client or a local vector store setup
# For demonstration, we'll use a simple in-memory mock or a file-based approach.
# In a real application, this would integrate with Pinecone, Milvus, Chroma, FAISS, etc.

# Import dependencies
from config.config_manager import config_manager
from utils.analytics_tracker import log_event
# CORRECTED: Import the CloudStorageUtilsWrapper class, not individual functions
from shared_tools.cloud_storage_utils import CloudStorageUtilsWrapper 
# Assuming you have a document loader/splitter (e.g., from langchain)
# For this example, we'll mock it.

logger = logging.getLogger(__name__)

# Base directory for local vector stores (if applicable, e.g., for FAISS)
BASE_VECTOR_DIR = Path("vector_stores")
os.makedirs(BASE_VECTOR_DIR, exist_ok=True)

class VectorUtilsWrapper:
    """
    A wrapper class for vector database operations, including document processing,
    embedding, storage, and retrieval.
    It integrates with CloudStorageUtilsWrapper for file handling and FirestoreManager
    for metadata storage.
    """
    def __init__(self, firestore_manager, cloud_storage_utils: CloudStorageUtilsWrapper, config_manager):
        self.firestore_manager = firestore_manager
        self.cloud_storage_utils = cloud_storage_utils # Store the instantiated wrapper
        self.config_manager = config_manager
        logger.info("VectorUtilsWrapper initialized.")

    async def process_uploaded_document(
        self,
        user_id: str,
        file_name: str,
        file_content_base64: str,
        # Managers are passed in from main.py during the FastAPI endpoint call
        firestore_manager: Any, # FirestoreManager instance
        cloud_storage_utils: CloudStorageUtilsWrapper, # CloudStorageUtilsWrapper instance
        config_manager: Any, # ConfigManager instance
        log_event_func: Any # log_event function
    ) -> Dict[str, Any]:
        """
        Handles the end-to-end process of uploading, storing, and indexing a document.

        Args:
            user_id (str): The ID of the user uploading the document.
            file_name (str): The original name of the uploaded file.
            file_content_base64 (str): The base64 encoded content of the file.
            firestore_manager: The FirestoreManager instance.
            cloud_storage_utils: The CloudStorageUtilsWrapper instance.
            config_manager: The ConfigManager instance.
            log_event_func: The log_event function from analytics_tracker.

        Returns:
            Dict[str, Any]: A dictionary indicating success, message, and document_id.
        """
        logger.info(f"Processing uploaded document: {file_name} for user: {user_id}")
        
        # 1. Upload file to cloud storage
        upload_result = await cloud_storage_utils.upload_file_to_storage(user_id, file_name, file_content_base64)
        if not upload_result["success"]:
            await log_event_func('document_processing', {
                'operation': 'upload_to_gcs',
                'file_name': file_name,
                'status': 'failed',
                'error': upload_result.get('message')
            }, user_id=user_id, success=False)
            return {"success": False, "message": f"Failed to upload document to cloud storage: {upload_result.get('message')}"}
        
        gcs_file_url = upload_result["file_url"]
        logger.info(f"Document uploaded to GCS: {gcs_file_url}")

        # 2. Store document metadata in Firestore
        document_metadata = {
            "user_id": user_id,
            "file_name": file_name,
            "gcs_url": gcs_file_url,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            "status": "uploaded",
            "indexed": False # Will be set to True after vector indexing
        }
        try:
            doc_ref = await firestore_manager.add_document_metadata(user_id, document_metadata)
            document_id = doc_ref.id
            logger.info(f"Document metadata stored in Firestore with ID: {document_id}")
            await log_event_func('document_processing', {
                'operation': 'store_metadata',
                'file_name': file_name,
                'document_id': document_id,
                'status': 'success'
            }, user_id=user_id, success=True)
        except Exception as e:
            await log_event_func('document_processing', {
                'operation': 'store_metadata',
                'file_name': file_name,
                'status': 'failed',
                'error': str(e)
            }, user_id=user_id, success=False)
            # Attempt to delete from GCS if metadata storage fails
            await cloud_storage_utils.delete_file_from_storage(user_id, file_name)
            return {"success": False, "message": f"Failed to store document metadata: {e}"}

        # 3. Read content for processing (if needed for embedding/chunking)
        # For this example, we'll directly use the decoded content from base64
        file_content_bytes = base64.b64decode(file_content_base64)
        file_content_str = file_content_bytes.decode('utf-8', errors='ignore') # Decode to string

        # 4. Chunk and Embed Document (Placeholder for actual RAG pipeline)
        # In a real RAG system, you would:
        # a. Load the document content (already have file_content_str)
        # b. Split it into smaller chunks (e.g., using RecursiveCharacterTextSplitter)
        # c. Generate embeddings for each chunk (using an embedding model like OpenAI, Cohere, Sentence Transformers)
        # d. Store chunks and their embeddings in a vector database (e.g., Pinecone, Chroma, FAISS)

        # Mocking chunking and embedding for now
        mock_chunks = [
            f"Chunk 1 of {file_name}: {file_content_str[:100]}...",
            f"Chunk 2 of {file_name}: {file_content_str[100:200]}...",
        ]
        mock_embeddings = ["embedding_data_1", "embedding_data_2"] # Placeholder

        try:
            # Simulate storing in a vector DB
            # For a real vector DB, this would be client.upsert(vectors=...)
            logger.info(f"Simulating vector indexing for document ID: {document_id}")
            # Example: Save mock chunks/embeddings to a local file for testing
            mock_vector_store_path = BASE_VECTOR_DIR / user_id / f"{document_id}_vectors.json"
            mock_vector_store_path.parent.mkdir(parents=True, exist_ok=True)
            with open(mock_vector_store_path, "w") as f:
                json.dump({"chunks": mock_chunks, "embeddings": mock_embeddings}, f)
            
            await firestore_manager.update_document_metadata(user_id, document_id, {"status": "indexed", "indexed": True, "vector_store_path": str(mock_vector_store_path)})
            logger.info(f"Document {document_id} successfully indexed.")
            await log_event_func('document_processing', {
                'operation': 'vector_indexing',
                'document_id': document_id,
                'status': 'success',
                'num_chunks': len(mock_chunks)
            }, user_id=user_id, success=True)
            return {"success": True, "message": "Document uploaded and indexed successfully.", "document_id": document_id}
        except Exception as e:
            await log_event_func('document_processing', {
                'operation': 'vector_indexing',
                'document_id': document_id,
                'status': 'failed',
                'error': str(e)
            }, user_id=user_id, success=False)
            # If indexing fails, update status in Firestore and potentially clean up GCS upload
            await firestore_manager.update_document_metadata(user_id, document_id, {"status": "indexing_failed"})
            # Decide if you want to delete the file from GCS on indexing failure
            # await cloud_storage_utils.delete_file_from_storage(user_id, file_name)
            return {"success": False, "message": f"Failed to index document: {e}"}

    async def query_uploaded_docs(
        self,
        query_text: str,
        user_token: str,
        export: Optional[bool] = False,
        k: int = 5
    ) -> str:
        """
        Queries previously uploaded and indexed documents for a user using vector similarity search.
        
        Args:
            query_text (str): The search query to find relevant documents.
            user_token (str): The unique identifier for the user.
            export (bool): If True, the results will be saved to a file in markdown format. Defaults to False.
            k (int): The number of top relevant documents to retrieve. Defaults to 5.
        
        Returns:
            str: A string containing the combined content of the relevant document chunks,
                 or a message indicating no data/results found, or the export path if exported.
        """
        logger.info(f"Querying uploaded docs for user {user_token} with query: '{query_text}'")

        # In a real RAG system, you would:
        # 1. Generate embedding for query_text
        # 2. Perform similarity search against the user's vector store
        # 3. Retrieve top-k relevant chunks

        # Mocking document retrieval for now
        mock_document_id = "mock_doc_id_123" # Assume a mock document exists for the user
        mock_vector_store_path = BASE_VECTOR_DIR / user_token / f"{mock_document_id}_vectors.json"

        retrieved_content = []
        if mock_vector_store_path.exists():
            try:
                with open(mock_vector_store_path, "r") as f:
                    mock_data = json.load(f)
                mock_chunks = mock_data.get("chunks", [])
                # Simulate finding relevant chunks
                retrieved_content = [chunk for chunk in mock_chunks if query_text.lower() in chunk.lower()]
                if not retrieved_content:
                    retrieved_content = mock_chunks[:k] # Just return first k if no specific match
            except Exception as e:
                logger.error(f"Error loading mock vector store for user {user_token}: {e}")
                retrieved_content = [f"Error retrieving mock content: {e}"]
        else:
            retrieved_content = ["No indexed documents found for this user (mock)."]

        combined_content = "\n\n".join(retrieved_content)

        if export:
            export_dir = Path("exports") / user_token
            export_dir.mkdir(parents=True, exist_ok=True)
            export_file_path = export_dir / f"query_results_{datetime.now().strftime('%Y%m%d%H%M%S')}.md"
            with open(export_file_path, "w") as f:
                f.write(f"# Query Results for '{query_text}'\n\n")
                f.write(combined_content)
            logger.info(f"Query results exported to {export_file_path}")
            await log_event('document_query', {
                'query': query_text,
                'status': 'exported',
                'export_path': str(export_file_path)
            }, user_id=user_token, success=True)
            return f"Query results exported to: {export_file_path}"
        else:
            await log_event('document_query', {
                'query': query_text,
                'status': 'retrieved',
                'num_results': len(retrieved_content)
            }, user_id=user_token, success=True)
            return combined_content if combined_content else "No relevant information found in uploaded documents."

# CLI Test (optional)
if __name__ == "__main__":
    from unittest.mock import MagicMock, patch, AsyncMock
    import shutil
    from datetime import datetime, timezone

    logging.basicConfig(level=logging.INFO)

    # Mock ConfigManager for CLI tests
    class MockConfigManager:
        def get(self, key, default=None):
            if key == "cloud_storage_bucket_name":
                return "mock-test-bucket"
            return default
        def get_secret(self, key, default=None):
            if key == "gcs_bucket_name": # Used by CloudStorageUtilsWrapper
                return "mock-test-bucket"
            return default

    # Mock FirestoreManager
    class MockFirestoreManager:
        async def add_document_metadata(self, user_id, metadata):
            logger.info(f"MockFirestoreManager: Adding metadata for {user_id}: {metadata['file_name']}")
            mock_doc_ref = MagicMock()
            mock_doc_ref.id = "mock_doc_id_123"
            return mock_doc_ref
        
        async def update_document_metadata(self, user_id, doc_id, updates):
            logger.info(f"MockFirestoreManager: Updating metadata for {doc_id}: {updates}")
            return True

        async def get_document_metadata(self, user_id, doc_id):
            logger.info(f"MockFirestoreManager: Getting metadata for {doc_id}")
            return {"user_id": user_id, "file_name": "mock_file.pdf", "gcs_url": "gs://mock/mock_file.pdf"}

    # Mock CloudStorageUtilsWrapper
    class MockCloudStorageUtilsWrapper:
        def __init__(self, config_manager_instance):
            self.config_manager = config_manager_instance
            logger.info("MockCloudStorageUtilsWrapper initialized.")

        async def upload_file_to_storage(self, user_id, file_name, file_content_base64):
            logger.info(f"MockCloudStorageUtilsWrapper: Uploading {file_name} for {user_id}")
            return {"success": True, "message": "Mock upload success", "file_url": f"gs://mock-bucket/{user_id}/{file_name}"}

        async def download_file_from_storage(self, user_id, file_name):
            logger.info(f"MockCloudStorageUtilsWrapper: Downloading {file_name} for {user_id}")
            return {"success": True, "message": "Mock download success", "content": "mock file content"}

        async def delete_file_from_storage(self, user_id, file_name):
            logger.info(f"MockCloudStorageUtilsWrapper: Deleting {file_name} for {user_id}")
            return {"success": True, "message": "Mock delete success"}

        async def read_file_from_storage_to_bytes(self, user_id, file_name):
            logger.info(f"MockCloudStorageUtilsWrapper: Reading {file_name} for {user_id} to bytes")
            return {"success": True, "message": "Mock read success", "content": b"mock file content"}


    # Mock log_event function
    mock_log_event = AsyncMock()

    async def run_vector_utils_tests():
        print("\n--- Testing VectorUtilsWrapper ---")
        mock_firestore = MockFirestoreManager()
        mock_cloud_storage = MockCloudStorageUtilsWrapper(MockConfigManager())
        mock_config = MockConfigManager()

        vector_utils_instance = VectorUtilsWrapper(
            firestore_manager=mock_firestore,
            cloud_storage_utils=mock_cloud_storage,
            config_manager=mock_config
        )

        test_user_id = "test_user_abc"
        test_file_name = "my_report.txt"
        test_content = "This is the content of my important report. It has several key points."
        test_content_base64 = base64.b64encode(test_content.encode('utf-8')).decode('utf-8')

        # Clean up existing mock vector store directory if it exists
        test_vector_dir = BASE_VECTOR_DIR / test_user_id
        if test_vector_dir.exists():
            shutil.rmtree(test_vector_dir)
            logger.info(f"Cleaned up {test_vector_dir}")

        # Test process_uploaded_document
        print("\n--- Test 1: process_uploaded_document (Success) ---")
        result = await vector_utils_instance.process_uploaded_document(
            user_id=test_user_id,
            file_name=test_file_name,
            file_content_base64=test_content_base64,
            firestore_manager=mock_firestore,
            cloud_storage_utils=mock_cloud_storage,
            config_manager=mock_config,
            log_event_func=mock_log_event
        )
        print(f"Process Uploaded Document Result: {result}")
        assert result["success"] is True
        assert "Document uploaded and indexed successfully." in result["message"]
        assert result["document_id"] == "mock_doc_id_123"
        # Verify mock_log_event was called for upload and indexing
        mock_log_event.assert_any_call('cloud_storage_operation', Any, user_id=test_user_id, success=True)
        mock_log_event.assert_any_call('document_processing', {'operation': 'store_metadata', 'file_name': test_file_name, 'document_id': 'mock_doc_id_123', 'status': 'success'}, user_id=test_user_id, success=True)
        mock_log_event.assert_any_call('document_processing', {'operation': 'vector_indexing', 'document_id': 'mock_doc_id_123', 'status': 'success', 'num_chunks': 2}, user_id=test_user_id, success=True)
        mock_log_event.reset_mock()
        print("Test 1 Passed.")

        # Test query_uploaded_docs
        print("\n--- Test 2: query_uploaded_docs (Success) ---")
        query_result = await vector_utils_instance.query_uploaded_docs(
            query_text="key points",
            user_token=test_user_id
        )
        print(f"Query Uploaded Docs Result: {query_result}")
        assert "key points" in query_result
        mock_log_event.assert_called_once()
        args, kwargs = mock_log_event.call_args
        logged_data = args[0]
        assert logged_data == 'document_query'
        assert kwargs['success'] is True
        mock_log_event.reset_mock()
        print("Test 2 Passed.")

        # Test query_uploaded_docs (No relevant info)
        print("\n--- Test 3: query_uploaded_docs (No relevant info) ---")
        query_result_no_info = await vector_utils_instance.query_uploaded_docs(
            query_text="nonexistent keyword",
            user_token=test_user_id
        )
        print(f"Query Uploaded Docs Result (No Info): {query_result_no_info}")
        assert "No relevant information found in uploaded documents." in query_result_no_info
        mock_log_event.assert_called_once()
        mock_log_event.reset_mock()
        print("Test 3 Passed.")

        # Test query_uploaded_docs (Export)
        print("\n--- Test 4: query_uploaded_docs (Export) ---")
        export_result = await vector_utils_instance.query_uploaded_docs(
            query_text="report",
            user_token=test_user_id,
            export=True
        )
        print(f"Query Uploaded Docs Result (Export): {export_result}")
        assert "Query results exported to:" in export_result
        export_path = Path(export_result.replace("Query results exported to: ", ""))
        assert export_path.exists()
        with open(export_path, "r") as f:
            exported_content = f.read()
            assert "key points" in exported_content
            assert "report" in exported_content
        mock_log_event.assert_called_once()
        mock_log_event.reset_mock()
        print("Test 4 Passed.")
        
        # Clean up exported file
        if export_path.exists():
            os.remove(export_path)

        # Clean up mock vector store directory
        if test_vector_dir.exists():
            shutil.rmtree(test_vector_dir)
            logger.info(f"Cleaned up {test_vector_dir}")

        print("\nAll VectorUtilsWrapper tests completed.")

    if __name__ == "__main__":
        asyncio.run(run_vector_utils_tests())


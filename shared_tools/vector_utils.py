# shared_tools/vector_utils.py

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import os
import shutil
import base64
import asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock # For mocks

# Import necessary components for vector storage and processing
# For demonstration, we'll use a simple in-memory mock or a file-based approach.
# In a real application, this would integrate with Pinecone, Milvus, Chroma, FAISS, etc.

# Import dependencies
from config.config_manager import config_manager
from utils.analytics_tracker import log_event
# CORRECTED: Import the CloudStorageUtilsWrapper class
from shared_tools.cloud_storage_utils import CloudStorageUtilsWrapper 

logger = logging.getLogger(__name__)

# Base directory for local vector stores (if applicable, e.g., for FAISS)
BASE_VECTOR_DIR = Path("vector_stores")
os.makedirs(BASE_VECTOR_DIR, exist_ok=True)

# --- Mocks for missing dependencies (replace with actual implementations) ---
class MockFirestoreManager:
    """Mock FirestoreManager for testing purposes."""
    def __init__(self):
        logger.info("Using MockFirestoreManager")
    async def get_document_by_id(self, collection_path, document_id):
        logger.debug(f"MockFirestoreManager: Getting doc {document_id} from {collection_path}")
        return {"status": "mock_success", "data": f"Mock doc for {document_id}"}
    async def set_document(self, collection_path, document_id, data):
        logger.debug(f"MockFirestoreManager: Setting doc {document_id} in {collection_path}")
        return {"status": "mock_success"}
    async def collection(self, collection_path):
        """Allows chaining like db.collection('name').document('id') or .where()..."""
        logger.debug(f"MockFirestoreManager: Accessing collection {collection_path}")
        return self # Return self to allow method chaining
    async def document(self, document_id):
        """Allows chaining for specific documents."""
        logger.debug(f"MockFirestoreManager: Accessing document {document_id}")
        return self # Return self to allow method chaining
    async def get(self):
        """Mocks the .get() call on a document reference."""
        logger.debug("MockFirestoreManager: Performing mock .get()")
        return MagicMock(exists=True, to_dict=lambda: {"mock_field": "mock_value"})
    async def add(self, data):
        logger.debug(f"MockFirestoreManager: Adding document with data {data}")
        return MagicMock(id="mock_doc_id")

class MockCloudStorageUtilsWrapper:
    """Mock CloudStorageUtilsWrapper for testing when GCS is not configured."""
    def __init__(self, config_manager):
        logger.info("Using MockCloudStorageUtilsWrapper")
        self.config_manager = config_manager # Keep it for consistency, but not used by mocks
        self.bucket_name = "mock-bucket"

    async def upload_file(self, source_file_path: Path, destination_blob_name: str) -> str:
        logger.info(f"Mock GCS: Uploading {source_file_path} to {destination_blob_name}")
        return f"gs://{self.bucket_name}/{destination_blob_name}"

    async def download_file(self, source_blob_name: str, destination_file_path: Path) -> str:
        logger.info(f"Mock GCS: Downloading {source_blob_name} to {destination_file_path}")
        destination_file_path.touch() # Create an empty file to simulate download
        return str(destination_file_path)

    async def delete_blob(self, blob_name: str) -> str:
        logger.info(f"Mock GCS: Deleting {blob_name}")
        return f"Blob {blob_name} deleted."


# Determine which CloudStorageUtilsWrapper to use based on config or env
# Check if a GCS bucket name is configured in config_manager
# CORRECTED: Changed get_config to get to match ConfigManager's method signature
gcs_bucket_configured = config_manager.get("gcs_bucket_name") is not None

if gcs_bucket_configured:
    # Attempt to use the real CloudStorageUtilsWrapper
    # Its __init__ will log errors if GCS credentials are bad, but won't crash
    _cloud_storage_utils_instance = CloudStorageUtilsWrapper(config_manager)
    # If GCS client failed to initialize within CloudStorageUtilsWrapper,
    # _cloud_storage_utils_instance._gcs_client will be None.
    # We can further check it here if we want to switch to mock post-init.
    if _cloud_storage_utils_instance._gcs_client is None:
        logger.warning("Real GCS client failed to initialize, falling back to MockCloudStorageUtilsWrapper.")
        _cloud_storage_utils_instance = MockCloudStorageUtilsWrapper(config_manager)
else:
    logger.info("GCS bucket name not configured. Using MockCloudStorageUtilsWrapper.")
    _cloud_storage_utils_instance = MockCloudStorageUtilsWrapper(config_manager)

# Global instance of VectorUtilsWrapper, initialized lazily
_vector_utils_instance: Optional['VectorUtilsWrapper'] = None

def _get_vector_utils_instance() -> 'VectorUtilsWrapper':
    """
    Returns a lazily initialized singleton instance of VectorUtilsWrapper.
    """
    global _vector_utils_instance
    if _vector_utils_instance is None:
        _vector_utils_instance = VectorUtilsWrapper(
            firestore_manager=MockFirestoreManager(), # Replace with your actual FirestoreManager instance
            cloud_storage_utils=_cloud_storage_utils_instance, # This will be either real or mock GCS
            config_manager=config_manager
        )
    return _vector_utils_instance

class VectorUtilsWrapper:
    """
    A wrapper class for vector database operations, including document processing,
    embedding, storage, and retrieval.
    It integrates with CloudStorageUtilsWrapper for file handling and FirestoreManager
    for metadata storage.
    """
    def __init__(self, firestore_manager, cloud_storage_utils: CloudStorageUtilsWrapper, config_manager):
        self.firestore_manager = firestore_manager
        self.cloud_storage_utils = cloud_storage_utils
        self.config_manager = config_manager
        logger.info("VectorUtilsWrapper initialized.")

    async def initialize_vector_store(self, user_id: str, section: str):
        """
        Initializes or ensures the existence of the vector store for a given user and section.
        This might involve creating directories, setting up database connections, etc.
        """
        # For a FAISS-like local store:
        vector_store_path = BASE_VECTOR_DIR / user_id / section
        vector_store_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized vector store path: {vector_store_path}")
        # In a real scenario, this would set up connections to Pinecone, Chroma, etc.

    async def add_document_to_vector_store(self, user_id: str, section: str, document_content: str, metadata: Dict[str, Any]) -> str:
        """
        Processes a document, generates embeddings, and adds it to the vector store.
        """
        await self.initialize_vector_store(user_id, section) # Ensure store is ready

        # Mocking document processing and embedding
        doc_id = f"{section}_{os.urandom(4).hex()}" # Simple mock ID
        # In a real implementation:
        # 1. Load document (e.g., using langchain document loaders)
        # 2. Split into chunks (e.g., RecursiveCharacterTextSplitter)
        # 3. Generate embeddings for each chunk (e.g., OpenAIEmbeddings, HuggingFaceEmbeddings)
        # 4. Add chunks and embeddings to vector database (e.g., FAISS.from_documents, Pinecone.from_documents)
        logger.info(f"Mock: Adding document {doc_id} to vector store for user {user_id}, section {section}")
        
        # Mock storing metadata in Firestore
        doc_metadata = {
            "user_id": user_id,
            "section": section,
            "document_id": doc_id,
            "content_snippet": document_content[:200] + "..." if len(document_content) > 200 else document_content,
            "metadata": metadata,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        await self.firestore_manager.collection("vector_store_metadata").document(doc_id).set_document(doc_metadata)

        log_event(user_id, "add_document_to_vector_store", "success", {"section": section, "document_id": doc_id})
        return f"Document {doc_id} added to vector store for section '{section}'."

    async def load_vectorstore(self, user_id: str, section: str) -> Any:
        """
        Loads the appropriate vector store for a given user and section.
        In a real application, this would load/connect to the specific vector DB index/collection.
        For now, it returns a mock object that simulates search.
        """
        await self.initialize_vector_store(user_id, section) # Ensure path exists

        # Mock vector store for demonstration
        class MockVectorStore:
            def similarity_search(self, query: str, k: int) -> List[Any]:
                logger.info(f"MockVectorStore: Performing similarity search for '{query}' (k={k})")
                if "report" in query.lower():
                    # Simulate finding a specific document
                    doc_content = "This is a mock report with key points about financial performance."
                    return [MagicMock(page_content=doc_content, metadata={"source": "mock_report.pdf"})]
                elif "no info" in query.lower():
                    return []
                # Simulate general results
                docs = [
                    MagicMock(page_content=f"Relevant data point 1 about {query}.", metadata={"source": f"{section}_doc_a.txt"}),
                    MagicMock(page_content=f"Key insight 2 related to {query}.", metadata={"source": f"{section}_doc_b.txt"})
                ]
                return docs[:k] # Return up to k mock documents

        logger.info(f"Returning mock vector store for user '{user_id}' in section '{section}'.")
        log_event(user_id, "load_vectorstore", "success", {"section": section})
        return MockVectorStore()

    async def delete_vector_store(self, user_id: str, section: str) -> str:
        """
        Deletes the vector store for a given user and section.
        """
        vector_store_path = BASE_VECTOR_DIR / user_id / section
        if vector_store_path.exists() and vector_store_path.is_dir():
            shutil.rmtree(vector_store_path)
            logger.info(f"Deleted vector store directory: {vector_store_path}")
            log_event(user_id, "delete_vector_store", "success", {"section": section})
            return f"Vector store for section '{section}' deleted."
        log_event(user_id, "delete_vector_store", "not_found", {"section": section})
        return f"No vector store found for section '{section}' to delete."

    async def list_indexed_documents(self, user_id: str, section: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Lists metadata for documents indexed in the vector store for a user,
        optionally filtered by section.
        """
        query_ref = self.firestore_manager.collection("vector_store_metadata").where("user_id", "==", user_id)
        if section:
            query_ref = query_ref.where("section", "==", section)
        
        # Mocking the query results
        mock_docs = [
            {"user_id": user_id, "document_id": "doc1", "section": "general", "content_snippet": "First document...", "metadata": {"title": "Doc One"}},
            {"user_id": user_id, "document_id": "doc2", "section": "finance", "content_snippet": "Finance report summary...", "metadata": {"title": "Financials"}},
        ]
        results = [d for d in mock_docs if d["user_id"] == user_id and (not section or d["section"] == section)]

        log_event(user_id, "list_indexed_documents", "success", {"section": section, "count": len(results)})
        return results

# Module-level function for load_vectorstore, wrapping the instance method
async def load_vectorstore(user_id: str, section: str) -> Any:
    """
    Loads the appropriate vector store for a given user and section.
    This is a module-level function that wraps the VectorUtilsWrapper method.
    """
    instance = _get_vector_utils_instance()
    return await instance.load_vectorstore(user_id, section)

# Module-level function for add_document_to_vector_store, wrapping the instance method
async def add_document_to_vector_store(user_id: str, section: str, document_content: str, metadata: Dict[str, Any]) -> str:
    """
    Processes a document, generates embeddings, and adds it to the vector store.
    This is a module-level function that wraps the VectorUtilsWrapper method.
    """
    instance = _get_vector_utils_instance()
    return await instance.add_document_to_vector_store(user_id, section, document_content, metadata)

# Module-level function for delete_vector_store, wrapping the instance method
async def delete_vector_store(user_id: str, section: str) -> str:
    """
    Deletes the vector store for a given user and section.
    This is a module-level function that wraps the VectorUtilsWrapper method.
    """
    instance = _get_vector_utils_instance()
    return await instance.delete_vector_store(user_id, section)

# Module-level function for list_indexed_documents, wrapping the instance method
async def list_indexed_documents(user_id: str, section: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Lists metadata for documents indexed in the vector store for a user,
    optionally filtered by section.
    This is a module-level function that wraps the VectorUtilsWrapper method.
    """
    instance = _get_vector_utils_instance()
    return await instance.list_indexed_documents(user_id, section)


# CLI Test (optional)
if __name__ == "__main__":
    from unittest.mock import MagicMock
    import sys
    import asyncio
    
    logging.basicConfig(level=logging.INFO)

    # Mock config_manager for testing purposes
    class MockConfigManager:
        def __init__(self):
            self._secrets = {}
            # Set to None by default for mock GCS if no specific bucket name is provided
            self._configs = {"gcs_bucket_name": None} 

        def get_secret(self, key: str) -> Optional[str]:
            return self._secrets.get(key)

        def set_secret(self, key: str, value: str):
            self._secrets[key] = value

        def get(self, key: str, default: Any = None) -> Any: # Changed get_config to get
            return self._configs.get(key, default)
        
        def set_config(self, key: str, value: Any):
            self._configs[key] = value

    sys.modules['config.config_manager'] = MockConfigManager()
    sys.modules['utils.analytics_tracker'] = MagicMock()
    sys.modules['utils.analytics_tracker'].log_event = MagicMock()

    # Create a test config manager
    test_config_manager = MockConfigManager()
    # If you want to test with real GCS (after setting up credentials), uncomment below:
    # test_config_manager.set_config("gcs_bucket_name", "your-test-bucket-name")
    # os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/path/to/your/laxjovial-c126e7532150.json" # Set this if you have the file

    # Note: For module-level imports, the global config_manager is used at import time.
    # To properly test the GCS fallback logic in `if __name__ == "__main__":` block,
    # you might need to run this file directly and ensure `gcs_bucket_configured` evaluates
    # based on the `test_config_manager` if it's assigned to the global `config_manager`
    # *before* `_cloud_storage_utils_instance` is initialized.
    # For now, the existing `if gcs_bucket_configured:` logic will handle it based on the
    # `config_manager` that was imported at the top of the module.

    async def run_vector_utils_tests():
        print("Running VectorUtilsWrapper tests...")

        test_user_id = "test_user_123"
        test_section = "general"
        test_document_content = "This is a sample document content for testing purposes. It contains some keywords like 'report' and 'data analysis'."
        test_metadata = {"source": "test_file.txt", "pages": 1}

        # Clean up existing test store
        test_vector_dir = BASE_VECTOR_DIR / test_user_id / test_section
        if test_vector_dir.exists():
            shutil.rmtree(test_vector_dir)

        # Test add_document_to_vector_store
        print("\n--- Test 1: add_document_to_vector_store ---")
        add_result = await add_document_to_vector_store(
            user_id=test_user_id,
            section=test_section,
            document_content=test_document_content,
            metadata=test_metadata
        )
        print(f"Add Document Result: {add_result}")
        assert f"Document {test_section}_" in add_result and "added to vector store" in add_result
        assert test_vector_dir.exists()
        sys.modules['utils.analytics_tracker'].log_event.assert_called_once()
        sys.modules['utils.analytics_tracker'].log_event.reset_mock()
        print("Test 1 Passed.")

        # Test load_vectorstore and similarity_search
        print("\n--- Test 2: load_vectorstore and similarity_search ---")
        vectorstore = await load_vectorstore(user_id=test_user_id, section=test_section)
        assert vectorstore is not None

        query_result = vectorstore.similarity_search("sample query", k=1)
        print(f"Query Result: {query_result[0].page_content}")
        assert "Relevant data point 1" in query_result[0].page_content
        sys.modules['utils.analytics_tracker'].log_event.assert_called_once()
        sys.modules['utils.analytics_tracker'].log_event.reset_mock()
        print("Test 2 Passed.")

        # Test list_indexed_documents
        print("\n--- Test 3: list_indexed_documents ---")
        indexed_docs = await list_indexed_documents(user_id=test_user_id, section=test_section)
        print(f"Indexed Documents: {indexed_docs}")
        assert len(indexed_docs) > 0 # Should at least see the mock docs
        sys.modules['utils.analytics_tracker'].log_event.assert_called_once()
        sys.modules['utils.analytics_tracker'].log_event.reset_mock()
        print("Test 3 Passed.")


        # Test delete_vector_store
        print("\n--- Test 4: delete_vector_store ---")
        delete_result = await delete_vector_store(user_id=test_user_id, section=test_section)
        print(f"Delete Result: {delete_result}")
        assert "deleted" in delete_result
        assert not test_vector_dir.exists()
        sys.modules['utils.analytics_tracker'].log_event.assert_called_once()
        sys.modules['utils.analytics_tracker'].log_event.reset_mock()
        print("Test 4 Passed.")

        # Test GCS Mock functionality
        print("\n--- Test 5: GCS Mock (internal to VectorUtilsWrapper) ---")
        gcs_mock_instance = _get_vector_utils_instance().cloud_storage_utils
        assert isinstance(gcs_mock_instance, MockCloudStorageUtilsWrapper)

        mock_file = Path("mock_file.txt")
        mock_file.write_text("test content")
        upload_res = await gcs_mock_instance.upload_file(mock_file, "test_blob.txt")
        print(f"Mock GCS Upload Result: {upload_res}")
        assert "gs://mock-bucket/test_blob.txt" in upload_res
        mock_file.unlink() # Clean up

        download_res = await gcs_mock_instance.download_file("test_blob.txt", Path("downloaded_mock.txt"))
        print(f"Mock GCS Download Result: {download_res}")
        assert "downloaded_mock.txt" in download_res
        Path("downloaded_mock.txt").unlink() # Clean up

        delete_res = await gcs_mock_instance.delete_blob("test_blob.txt")
        print(f"Mock GCS Delete Result: {delete_res}")
        assert "deleted" in delete_res
        print("Test 5 Passed.")


        print("\nAll VectorUtilsWrapper internal tests completed.")

    asyncio.run(run_vector_utils_tests())

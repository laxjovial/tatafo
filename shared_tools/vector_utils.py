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

# Assuming CloudStorageUtilsWrapper can be initialized with config_manager
_cloud_storage_utils_instance = CloudStorageUtilsWrapper(config_manager)

# --- Global instance of VectorUtilsWrapper and a factory function ---
_vector_utils_instance: Optional['VectorUtilsWrapper'] = None

def _get_vector_utils_instance() -> 'VectorUtilsWrapper':
    """
    Returns a lazily initialized singleton instance of VectorUtilsWrapper.
    In a production environment, ensure FirestoreManager and CloudStorageUtilsWrapper
    are properly initialized and passed here.
    """
    global _vector_utils_instance
    if _vector_utils_instance is None:
        _vector_utils_instance = VectorUtilsWrapper(
            firestore_manager=MockFirestoreManager(), # Replace with your actual FirestoreManager instance
            cloud_storage_utils=_cloud_storage_utils_instance, # Replace with your actual CloudStorageUtilsWrapper instance
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

        # In a real application, this would load the FAISS index, or connect to Pinecone/Chroma.
        # Example for FAISS:
        # try:
        #     embeddings = OpenAIEmbeddings() # Or your chosen embeddings
        #     vectorstore = FAISS.load_local(BASE_VECTOR_DIR / user_id / section, embeddings)
        # except Exception as e:
        #     logger.warning(f"Could not load local FAISS index for {user_id}/{section}: {e}")
        #     # Fallback or re-create
        #     vectorstore = FAISS.from_documents([], embeddings) # Create an empty one
        # return vectorstore

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
        # In a real scenario, you'd fetch from Firestore
        mock_docs = [
            {"document_id": "doc1", "section": "general", "content_snippet": "First document...", "metadata": {"title": "Doc One"}},
            {"document_id": "doc2", "section": "finance", "content_snippet": "Finance report summary...", "metadata": {"title": "Financials"}},
        ]
        results = [d for d in mock_docs if d["user_id"] == user_id and (not section or d["section"] == section)]

        log_event(user_id, "list_indexed_documents", "success", {"section": section, "count": len(results)})
        return results

# NEW: Module-level function for load_vectorstore, wrapping the instance method
async def load_vectorstore(user_id: str, section: str) -> Any:
    """
    Loads the appropriate vector store for a given user and section.
    This is a module-level function that wraps the VectorUtilsWrapper method.
    """
    instance = _get_vector_utils_instance()
    return await instance.load_vectorstore(user_id, section)

# NEW: Module-level function for add_document_to_vector_store, wrapping the instance method
async def add_document_to_vector_store(user_id: str, section: str, document_content: str, metadata: Dict[str, Any]) -> str:
    """
    Processes a document, generates embeddings, and adds it to the vector store.
    This is a module-level function that wraps the VectorUtilsWrapper method.
    """
    instance = _get_vector_utils_instance()
    return await instance.add_document_to_vector_store(user_id, section, document_content, metadata)

# NEW: Module-level function for delete_vector_store, wrapping the instance method
async def delete_vector_store(user_id: str, section: str) -> str:
    """
    Deletes the vector store for a given user and section.
    This is a module-level function that wraps the VectorUtilsWrapper method.
    """
    instance = _get_vector_utils_instance()
    return await instance.delete_vector_store(user_id, section)

# NEW: Module-level function for list_indexed_documents, wrapping the instance method
async def list_indexed_documents(user_id: str, section: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Lists metadata for documents indexed in the vector store for a user,
    optionally filtered by section.
    This is a module-level function that wraps the VectorUtilsWrapper method.
    """
    instance = _get_vector_utils_instance()
    return await instance.list_indexed_documents(user_id, section)


# CLI Test (optional) - Moved to query_uploaded_docs_tool.py if it's the primary tool to test
# To run tests for vector_utils.py directly, you would need to set up mocks appropriately.
if __name__ == "__main__":
    from unittest.mock import MagicMock
    from datetime import datetime, timezone

    logging.basicConfig(level=logging.INFO)

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
        log_event.assert_called_once()
        log_event.reset_mock()
        print("Test 1 Passed.")

        # Test load_vectorstore and similarity_search
        print("\n--- Test 2: load_vectorstore and similarity_search ---")
        vectorstore = await load_vectorstore(user_id=test_user_id, section=test_section)
        assert vectorstore is not None

        query_result = vectorstore.similarity_search("sample query", k=1)
        print(f"Query Result: {query_result[0].page_content}")
        assert "Relevant data point 1" in query_result[0].page_content
        log_event.assert_called_once()
        log_event.reset_mock()
        print("Test 2 Passed.")

        # Test query_uploaded_docs (no relevant info) - this test is for the tool, not vector_utils itself
        # print("\n--- Test 3: query_uploaded_docs (no relevant info) ---")
        # # This would call the query_uploaded_docs from query_uploaded_docs_tool.py, not directly part of vector_utils.py tests
        # query_result_no_info = await query_uploaded_docs( # This assumes query_uploaded_docs is imported
        #     query="no info",
        #     user_token=test_user_id,
        #     section=test_section
        # )
        # print(f"Query Uploaded Docs Result (No Info): {query_result_no_info}")
        # assert "No relevant information found in uploaded documents." in query_result_no_info
        # log_event.assert_called_once()
        # log_event.reset_mock()
        # print("Test 3 Passed.")

        # Test list_indexed_documents
        print("\n--- Test 3: list_indexed_documents ---")
        indexed_docs = await list_indexed_documents(user_id=test_user_id, section=test_section)
        print(f"Indexed Documents: {indexed_docs}")
        assert len(indexed_docs) > 0 # Should at least see the mock docs
        log_event.assert_called_once()
        log_event.reset_mock()
        print("Test 3 Passed.")


        # Test delete_vector_store
        print("\n--- Test 4: delete_vector_store ---")
        delete_result = await delete_vector_store(user_id=test_user_id, section=test_section)
        print(f"Delete Result: {delete_result}")
        assert "deleted" in delete_result
        assert not test_vector_dir.exists()
        log_event.assert_called_once()
        log_event.reset_mock()
        print("Test 4 Passed.")

        print("\nAll VectorUtilsWrapper internal tests completed.")

    asyncio.run(run_vector_utils_tests())

# backend/api/docs_api.py

import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from pydantic import BaseModel, Field

# Project Imports
from backend.middleware.auth_middleware import get_current_user
from backend.models.user_models import UserProfile
from config.config_manager import config_manager
from utils.analytics_tracker import log_event
from database.firestore_manager import FirestoreManager
from shared_tools.cloud_storage_utils import CloudStorageUtilsWrapper
from shared_tools.vector_utils import VectorUtilsWrapper
from domain_tools.document_tools.document_tool import DocumentTools


logger = logging.getLogger(__name__)
router = APIRouter()

# --- Pydantic Models for Request/Response Bodies ---
class DocumentQuery(BaseModel):
    query_text: str = Field(..., description="The query string to search for within documents.")
    section: str = Field("general", description="The section of documents to query (e.g., 'medical', 'legal', 'finance').")
    k: Optional[int] = Field(5, description="The number of top relevant documents to retrieve.")
    export: bool = Field(False, description="Whether to export the query results.")

class DocumentDelete(BaseModel):
    document_id: str = Field(..., description="The ID of the document to delete (e.g., blob_name in GCS).")
    section: str = Field("general", description="The section the document belongs to.")

# --- Dependency Injection for DocumentTools and its components ---

# Dependency for CloudStorageUtilsWrapper (singleton via config_manager)
def get_cloud_storage_utils(
    cfg_manager: config_manager = Depends(config_manager)
) -> CloudStorageUtilsWrapper:
    return CloudStorageUtilsWrapper(cfg_manager)

# Dependency for FirestoreManager (singleton from main.py, or mocked in tests)
# This will be overridden in main.py's `app.dependency_overrides`
def get_firestore_manager_dependency() -> FirestoreManager:
    # This is a placeholder. In main.py, you'll set:
    # app.dependency_overrides[get_firestore_manager_dependency] = lambda: firestore_manager_instance
    raise NotImplementedError("FirestoreManager dependency must be overridden in main.py")

# Dependency for VectorUtilsWrapper
def get_vector_utils_wrapper(
    cfg_manager: config_manager = Depends(config_manager),
    cloud_storage_utils: CloudStorageUtilsWrapper = Depends(get_cloud_storage_utils),
    firestore_manager: FirestoreManager = Depends(get_firestore_manager_dependency)
) -> VectorUtilsWrapper:
    # Pass log_event function directly
    return VectorUtilsWrapper(
        config_manager_instance=cfg_manager,
        log_event_func=log_event,
        cloud_storage_utils_wrapper_instance=cloud_storage_utils,
        firestore_manager_instance=firestore_manager
    )

# Dependency for DocumentTools
def get_document_tools(
    vector_utils_wrapper: VectorUtilsWrapper = Depends(get_vector_utils_wrapper),
    cfg_manager: config_manager = Depends(config_manager),
    firestore_manager: FirestoreManager = Depends(get_firestore_manager_dependency),
    cloud_storage_utils: CloudStorageUtilsWrapper = Depends(get_cloud_storage_utils)
) -> DocumentTools:
    return DocumentTools(
        vector_utils_wrapper=vector_utils_wrapper,
        config_manager=cfg_manager,
        firestore_manager=firestore_manager,
        cloud_storage_utils=cloud_storage_utils,
        log_event_func=log_event # Pass the log_event function
    )


# --- API Endpoints ---

@router.get("/status")
async def get_docs_api_status():
    """Returns the status of the Document API."""
    return {"status": "Document API is running and healthy"}

@router.post("/upload", summary="Upload and process a document")
async def upload_document(
    file: UploadFile = File(..., description="The document file to upload."),
    section: str = Form("general", description="The section the document belongs to (e.g., 'legal', 'medical')."),
    current_user: UserProfile = Depends(get_current_user),
    doc_tools: DocumentTools = Depends(get_document_tools)
):
    """
    Uploads a document, stores it, and processes it for indexing (e.g., into a vector store).
    """
    user_id = current_user.user_id
    file_name = file.filename
    logger.info(f"User {user_id} attempting to upload and process file: {file_name} in section {section}")

    try:
        # Pass the file content directly as bytes or a file-like object
        file_content = await file.read() # Read content as bytes

        result = await doc_tools.process_uploaded_document(
            file_name=file_name,
            file_content_bytes=file_content, # Pass bytes
            user_context=current_user,
            collection_name=section # Use section as collection_name
        )

        if result.get("success"):
            await log_event(
                'document_upload_success',
                {'filename': file_name, 'section': section, 'details': result.get('message')},
                user_id=user_id,
                success=True,
                log_from_backend=True
            )
            return {"message": result.get("message"), "document_id": result.get("document_id")}
        else:
            error_msg = result.get("message", "Document processing failed.")
            await log_event(
                'document_upload_failure',
                {'filename': file_name, 'section': section, 'error': error_msg},
                user_id=user_id,
                success=False,
                error_message=error_msg,
                log_from_backend=True
            )
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_msg)

    except HTTPException:
        raise # Re-raise FastAPI HTTPExceptions
    except Exception as e:
        logger.error(f"Error uploading/processing document for user {user_id}: {e}", exc_info=True)
        error_msg = f"An unexpected error occurred during document upload: {str(e)}"
        await log_event(
            'document_upload_failure',
            {'filename': file_name, 'section': section, 'error': error_msg},
            user_id=user_id,
            success=False,
            error_message=error_msg,
            log_from_backend=True
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_msg)

@router.post("/query", summary="Query user's uploaded documents")
async def query_documents_endpoint(
    query_data: DocumentQuery,
    current_user: UserProfile = Depends(get_current_user),
    doc_tools: DocumentTools = Depends(get_document_tools)
):
    """
    Queries the user's uploaded and indexed documents within a specific section.
    """
    user_id = current_user.user_id
    logger.info(f"User {user_id} querying documents in section {query_data.section} with query: {query_data.query_text}")

    try:
        result = await doc_tools.query_uploaded_docs(
            query_text=query_data.query_text,
            user_context=current_user,
            collection_name=query_data.section, # Use section as collection_name
            k=query_data.k,
            export=query_data.export
        )

        await log_event(
            'document_query_success',
            {'query': query_data.query_text, 'section': query_data.section, 'results_summary': result[:200]}, # Log a snippet
            user_id=user_id,
            success=True,
            log_from_backend=True
        )
        return {"query_results": result}

    except Exception as e:
        logger.error(f"Error querying documents for user {user_id}: {e}", exc_info=True)
        error_msg = f"An error occurred while querying documents: {str(e)}"
        await log_event(
            'document_query_failure',
            {'query': query_data.query_text, 'section': query_data.section, 'error': error_msg},
            user_id=user_id,
            success=False,
            error_message=error_msg,
            log_from_backend=True
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_msg)

@router.delete("/delete/{document_id}", summary="Delete a specific document")
async def delete_document_endpoint(
    document_id: str, # This will typically be the GCS blob_name
    section: str = Form("general", description="The section the document belongs to (e.g., 'legal', 'medical')."),
    current_user: UserProfile = Depends(get_current_user),
    doc_tools: DocumentTools = Depends(get_document_tools)
):
    """
    Deletes a specific document by its ID (GCS blob name) from storage and the vector store.
    """
    user_id = current_user.user_id
    logger.info(f"User {user_id} attempting to delete document: {document_id} from section {section}")

    try:
        result = await doc_tools.delete_document(
            file_name=document_id, # DocumentTools's delete uses file_name
            user_context=current_user,
            collection_name=section # Section acts as collection name for vector store deletion
        )

        if result.get("success"):
            await log_event(
                'document_delete_success',
                {'document_id': document_id, 'section': section, 'details': result.get('message')},
                user_id=user_id,
                success=True,
                log_from_backend=True
            )
            return {"message": result.get("message")}
        else:
            error_msg = result.get("message", "Document deletion failed.")
            await log_event(
                'document_delete_failure',
                {'document_id': document_id, 'section': section, 'error': error_msg},
                user_id=user_id,
                success=False,
                error_message=error_msg,
                log_from_backend=True
            )
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_msg)

    except HTTPException:
        raise # Re-raise FastAPI HTTPExceptions
    except Exception as e:
        logger.error(f"Error deleting document for user {user_id}: {e}", exc_info=True)
        error_msg = f"An unexpected error occurred during document deletion: {str(e)}"
        await log_event(
            'document_delete_failure',
            {'document_id': document_id, 'section': section, 'error': error_msg},
            user_id=user_id,
            success=False,
            error_message=error_msg,
            log_from_backend=True
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_msg)


@router.get("/list", summary="List user's uploaded documents")
async def list_documents_endpoint(
    current_user: UserProfile = Depends(get_current_user),
    doc_tools: DocumentTools = Depends(get_document_tools)
):
    """
    Lists all documents uploaded by the current user.
    """
    user_id = current_user.user_id
    logger.info(f"User {user_id} requesting list of documents.")

    try:
        # Assuming DocumentTools has a method to list documents.
        # This method would typically query metadata stored in Firestore.
        documents_metadata = await doc_tools.list_user_documents(user_context=current_user)

        await log_event(
            'document_list_success',
            {'num_documents': len(documents_metadata)},
            user_id=user_id,
            success=True,
            log_from_backend=True
        )
        return {"documents": documents_metadata}

    except Exception as e:
        logger.error(f"Error listing documents for user {user_id}: {e}", exc_info=True)
        error_msg = f"An error occurred while listing documents: {str(e)}"
        await log_event(
            'document_list_failure',
            {'error': error_msg},
            user_id=user_id,
            success=False,
            error_message=error_msg,
            log_from_backend=True
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_msg)

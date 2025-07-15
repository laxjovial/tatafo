from fastapi import APIRouter, Depends, HTTPException, Request
from typing import Dict, Any

from backend.middleware.auth_middleware import get_current_user
from backend.services.llm_service import LLMService, get_llm_service_dependency
from backend.models.user_models import UserProfile

router = APIRouter()

@router.post("/run-tool")
async def run_tool_endpoint(
    tool_data: Dict[str, Any],
    current_user: UserProfile = Depends(get_current_user),
    llm_service: LLMService = Depends(get_llm_service_dependency),
):
    """
    Dynamically runs a tool by name with the provided arguments.
    """
    tool_name = tool_data.get("tool_name")
    tool_args = tool_data.get("tool_args", {})

    if not tool_name:
        raise HTTPException(status_code=400, detail="Tool name must be provided.")

    try:
        result = await llm_service.run_tool_by_name(
            tool_name=tool_name,
            tool_args=tool_args,
            user_profile=current_user,
        )
        return {"result": result, "success": True}
    except ValueError as e:
        # Tool not found
        raise HTTPException(status_code=404, detail=str(e))
    except PermissionError as e:
        # User does not have permission for the tool
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        # Any other unexpected error
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

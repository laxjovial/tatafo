from fastapi import FastAPI, Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader
from starlette.status import HTTP_403_FORBIDDEN
from domain_tools.finance_tools.finance_tool import finance_get_historical_stock_prices
from domain_tools.crypto_tools.crypto_tool import crypto_get_historical_crypto_price
from shared_tools.historical_data_tool import _make_dynamic_api_request_historical
from utils.user_manager import get_user_by_uid, get_user_by_api_key
from functools import wraps
from shared_tools.llm_pipeline import execute_pipeline
from utils.error_handler import handle_error
import logging

app = FastAPI()
router = app

API_KEY_HEADER = APIKeyHeader(name="Authorization")

def get_current_user(api_key: str = Depends(API_KEY_HEADER)):
    user = get_user_by_api_key(api_key.split(" ")[1])
    if user is None:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, detail="Could not validate credentials"
        )
    return user

@app.post("/api/run-tool/{tool_name}")
async def run_tool(tool_name: str, request: Request, user: dict = Depends(get_current_user)):
    """
    Runs a tool with the given parameters.
    """
    try:
        body = await request.json()

        if tool_name == "finance_get_historical_stock_prices":
            # Tier check
            if user['tier'] not in ['paid', 'premium']:
                raise HTTPException(status_code=403, detail="This tool is not available for your tier.")

        elif tool_name == "crypto_get_historical_crypto_price":
            # Tier check
            if user['tier'] not in ['paid', 'premium']:
                raise HTTPException(status_code=403, detail="This tool is not available for your tier.")

        elif tool_name == "shared_make_dynamic_api_request_historical":
            # Tier check
            if user['tier'] not in ['premium']:
                raise HTTPException(status_code=403, detail="This tool is not available for your tier.")
            return _make_dynamic_api_request_historical(**body)

        else:
            raise HTTPException(status_code=404, detail="Tool not found")
    except Exception as e:
        handle_error(e, "An unexpected error occurred while running the tool.")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")

@app.post("/api/assistant")
async def assistant(request: Request, user: dict = Depends(get_current_user)):
    """
    Provides a chat interface to interact with an AI assistant that can use the available tools.
    """
    try:
        body = await request.json()
        query = body.get("query")

        if not query:
            raise HTTPException(status_code=400, detail="Query not provided")

        # Tier check
        if user['tier'] == 'free':
            raise HTTPException(status_code=403, detail="AI assistant is not available for free tier users.")

        response = execute_pipeline(query)
        return {"response": response}
    except Exception as e:
        handle_error(e, "An unexpected error occurred in the AI assistant.")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")

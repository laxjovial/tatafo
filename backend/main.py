# backend/main.py

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from typing import Annotated

# Import API routers (will be created soon)
from backend.api import auth_api
from backend.api import user_api
# from backend.api import tool_api # Future: for exposing tools via API
# from backend.api import payment_api # Future: for payment processing
# from backend.api import admin_api # Future: for granular admin controls

# Import middleware (will be created soon)
# from backend.middleware.auth_middleware import verify_token_middleware

# Initialize FastAPI app
app = FastAPI(
    title="Unified AI Assistant Backend API",
    description="API for user management, AI tools, and core services.",
    version="0.1.0",
)

# Configure CORS (Cross-Origin Resource Sharing)
# This is crucial for allowing your Streamlit frontend (running on a different port/domain)
# to communicate with your FastAPI backend.
origins = [
    "http://localhost",
    "http://localhost:8501",  # Default Streamlit port
    "http://localhost:3000",  # Example for React/other frontend dev servers
    # Add your deployed Streamlit app URL here when deploying to production
    # e.g., "https://your-streamlit-app.streamlit.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(auth_api.router, prefix="/auth", tags=["Authentication"])
app.include_router(user_api.router, prefix="/users", tags=["User Management"])
# app.include_router(tool_api.router, prefix="/tools", tags=["AI Tools"])
# app.include_router(payment_api.router, prefix="/payments", tags=["Payments"])
# app.include_router(admin_api.router, prefix="/admin", tags=["Admin"])


@app.get("/")
async def read_root():
    """
    Root endpoint for the API.
    """
    return {"message": "Welcome to the Unified AI Assistant Backend API!"}

# You would run this FastAPI app using Uvicorn:
# uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

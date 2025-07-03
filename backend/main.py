# backend/main.py

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from typing import Annotated

# Import API routers
from backend.api import auth_api
from backend.api import user_api
from backend.api import tool_api # NEW: Import the tool_api router
# from backend.api import payment_api # Future: for payment processing
# from backend.api import admin_api # Future: for granular admin controls

# Import middleware dependencies (if applying at router level)
from backend.middleware.auth_middleware import get_current_active_user, get_current_admin_user

# Initialize FastAPI app
app = FastAPI(
    title="Unified AI Assistant Backend API",
    description="API for user management, AI tools, and core services.",
    version="0.1.0",
)

# Configure CORS (Cross-Origin Resource Sharing)
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
# NEW: Include the tool_api router, protected by authentication
app.include_router(tool_api.router, prefix="/tools", tags=["AI Tools"], dependencies=[Depends(get_current_active_user)])
# app.include_router(payment_api.router, prefix="/payments", tags=["Payments"], dependencies=[Depends(get_current_active_user)])
# app.include_router(admin_api.router, prefix="/admin", tags=["Admin"], dependencies=[Depends(get_current_admin_user)])


@app.get("/")
async def read_root():
    """
    Root endpoint for the API.
    """
    return {"message": "Welcome to the Unified AI Assistant Backend API!"}

# Example of a protected endpoint (requires any active user)
@app.get("/protected-test")
async def protected_test_route(current_user: Annotated[dict, Depends(get_current_active_user)]):
    return {"message": f"Hello {current_user['username']}, you accessed a protected route!"}

# Example of an admin-only protected endpoint
@app.get("/admin-protected-test")
async def admin_protected_test_route(current_user: Annotated[dict, Depends(get_current_admin_user)]):
    return {"message": f"Hello Admin {current_user['username']}, you accessed an admin-only route!"}

# You would run this FastAPI app using Uvicorn:
# uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# Backend

The backend is a FastAPI application that provides the API endpoints for the application. It is responsible for handling user authentication, user management, and tool execution.

## Directory Structure

- **api:** This directory contains the API endpoints for the application.
- **middleware:** This directory contains the middleware for the application.
- **models:** This directory contains the Pydantic models for the application.
- **services:** This directory contains the services for the application.
- **main.py:** This is the main entry point for the backend application.

## API Endpoints

The backend provides the following API endpoints:

- **/auth:** This endpoint is used for user authentication.
- **/user:** This endpoint is used for user management.
- **/admin:** This endpoint is used for admin-level operations.
- **/tools:** This endpoint is used for tool execution.

## Services

The backend provides the following services:

- **AdminService:** This service provides administrative functionalities such as user management, configuration updates, and analytics retrieval.
- **ApiUsageService:** This service manages API call limits, usage tracking, and dynamic distribution for default APIs.
- **LLMService:** This service manages interactions with Large Language Models and orchestrates tool usage.

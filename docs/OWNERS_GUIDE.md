# Owner's Guide: Intelli-Agent Application

## 1. Introduction

This document provides a comprehensive technical overview of the Intelli-Agent application. It is intended for project owners, lead developers, and system administrators who will be responsible for the deployment, maintenance, and long-term ownership of the application.

## 2. Application Architecture

The application is a modern web application with a decoupled frontend and backend.

### 2.1. Frontend

*   **Framework:** React.js (v18)
*   **Routing:** `react-router-dom` for client-side routing.
*   **Styling:** Standard CSS with a component-based approach. Each component has its own CSS file. Global styles are managed in `src/styles/global.css`.
*   **State Management:** A combination of React hooks (`useState`, `useEffect`, `useContext`) and a custom `useAuth` hook for managing authentication state.
*   **Structure:** The frontend follows a standard React project structure:
    *   `src/components`: Reusable UI components.
    *   `src/pages`: Top-level components for each route.
    *   `src/hooks`: Custom React hooks.
    *   `src/services`: Modules for interacting with external services (Firebase, Analytics).
    *   `src/config.js`: Centralized configuration.
    *   `public`: Static assets and the main `index.html` file.

### 2.2. Backend

*   **Framework:** FastAPI (Python)
*   **Authentication:** The backend uses Firebase Authentication. The frontend authenticates with Firebase, and the resulting ID token is sent to the backend for verification.
*   **Database:** Firestore is used as the primary database.
*   **API:** The backend provides a RESTful API for the frontend to consume.


### 2.3. Storage Management

*   **Tiered Storage Limits:** The application implements a tiered storage system. Storage limits for each tier are defined in `data/tiers.yaml`.
*   **Admin Overrides:** Administrators can set custom storage limits for individual users, which will override the tier default.
*   **Usage Tracking:** User storage usage is tracked in Firestore and updated when files are uploaded or deleted.

## 3. Key Technologies

*   **React.js:** For building the user interface.
*   **FastAPI:** For building the backend API.
*   **Firebase:** For authentication and database services.
*   **Webpack:** For bundling the frontend assets.
*   **Jest & React Testing Library:** For testing the frontend components.

## 4. Deployment

### 4.1. Frontend

The frontend is a standard React application that can be deployed to any static hosting provider (e.g., Vercel, Netlify, AWS S3).

1.  **Build the application:**
    ```bash
    npm run build
    ```
2.  **Deploy the `build` directory:** The contents of the `build` directory can be uploaded to the hosting provider.

### 4.2. Backend

The backend is a Python application that can be deployed to any platform that supports Python (e.g., Heroku, AWS Elastic Beanstalk, Google Cloud Run).

## 5. Maintenance

### 5.1. Updating Dependencies

Regularly update the dependencies for both the frontend and backend to ensure security and stability.

*   **Frontend:**
    ```bash
    npm update
    ```
*   **Backend:**
    ```bash
    pip install -r requirements.txt --upgrade
    ```

### 5.2. Monitoring

The application includes an analytics service that logs events to Firestore. This can be used to monitor user activity and identify potential issues. The admin dashboard provides a view of these analytics events.

### 5.3. Backups

Firestore provides automatic backups. It is recommended to configure a regular backup schedule in the Google Cloud Console.

## 6. Important Technical Considerations

*   **Environment Variables:** The `FASTAPI_BASE_URL` is configured in `.streamlit/secrets.toml` and made available to the frontend. For production, ensure this is set correctly.
*   **Firebase Configuration:** The Firebase configuration is also in `.streamlit/secrets.toml`.

*   **Environment Variables:** The `FASTAPI_BASE_URL` is currently hardcoded in `src/config.js`. For production, this should be replaced with an environment variable to allow for different environments (development, staging, production).
*   **Firebase Configuration:** The Firebase configuration is also included in the source code. This should be moved to environment variables for better security.

*   **Security:** Ensure that all API endpoints are properly secured and that user roles are enforced on the backend. The frontend provides a `PrivateRoute` component, but the backend should always be the source of truth for authorization.
*   **Scalability:** Both FastAPI and Firestore are highly scalable. As the application grows, you may need to optimize Firestore queries and add more backend instances.

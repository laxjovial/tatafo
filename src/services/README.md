# Services (`src/services`)

This directory contains modules for interacting with external APIs and services. By centralizing this logic, we can easily manage our application's dependencies on external services and reuse the same service logic across multiple components.

## Services

*   **`firebase.js`:** This module initializes and configures the Firebase application object, authentication service, and Firestore database. It provides a single point of access to these services for the rest of the application.
*   **`analytics.js`:** This module provides functions for logging analytics events to Firestore. It includes functions for initializing the analytics service and logging events.

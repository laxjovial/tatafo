# Intelli-Agent

Intelli-Agent is a powerful, extensible, and secure AI assistant designed to provide intelligent assistance across a wide range of domains. It is built with a modular architecture that allows for easy extension and customization.

## Features

- **AI Assistant:** A powerful AI assistant that can be used to perform a wide range of tasks.
- **Diverse Domain-specific tools:** A rich set of domain-specific tools that can be used to perform a wide range of tasks.
- **Comprehensive Analytics:** A comprehensive analytics system that can be used to track tool usage and other analytics.
- **Robust RBAC:** A robust role-based access control system that can be used to control access to the different tools and features.
- **Multi-factor Security:** A multi-factor security system that can be used to protect user accounts.
- **Dynamic UI Customization:** A dynamic UI customization system that can be used to customize the look and feel of the application.
- **AI-driven Unanswered Query Analysis & Tool Suggestion:** An AI-driven system that can be used to analyze unanswered queries and suggest new tools and functions.
- **User/Global External API Management with Tiered Limits and Overrides:** A system that allows users to manage their own external API configurations and for the administrator to manage global/default external API configurations.

## Architecture

The application is divided into three main layers:

- **Frontend:** The frontend is a React.js application that provides the user interface for the application.
- **Backend:** The backend is a FastAPI application that provides the API endpoints for the application.
- **Infrastructure & External Services:** The infrastructure and external services layer includes the authentication provider, cloud storage, and other external services.

## Getting Started

To get started with the application, you will need to have Python, Node.js, and npm/yarn installed on your system.

### Backend Setup

1. Create a virtual environment: `python3 -m venv venv`
2. Activate the virtual environment: `source venv/bin/activate`
3. Install the dependencies: `pip install -r requirements.txt`
4. Run the backend: `uvicorn backend.main:app --reload`

### Frontend Setup

1. Install the dependencies: `npm install`
2. Run the frontend: `npm start`

## Contributing

Contributions are welcome! Please read the [contributing guidelines](CONTRIBUTING.md) for more information.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

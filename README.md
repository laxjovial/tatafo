# Intelli-Agent: A Full-Stack AI Assistant

Welcome to the Intelli-Agent project! This is a powerful, AI-driven assistant platform built with a modern, full-stack architecture. This repository contains the complete codebase for both the React frontend and the FastAPI backend.

This project has been extensively refactored to follow best practices for code quality, maintainability, and scalability.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Application](#running-the-application)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Contributing](#contributing)

## Project Overview

Intelli-Agent is a web application that provides users with a versatile AI assistant. It features a secure authentication system, a user-friendly dashboard, and a powerful admin panel for managing the application.

## Features

*   **Secure User Authentication:** Users can register and log in securely.
*   **AI Assistant:** A conversational AI that can answer questions and perform tasks.
*   **User Profile Management:** Users can view and update their profile information.
*   **Admin Dashboard:** A comprehensive dashboard for administrators to monitor analytics and manage users.
*   **Role-Based Access Control:** The application distinguishes between regular users and administrators, with different levels of access.

## Technology Stack

*   **Frontend:** React, React Router, Standard CSS
*   **Backend:** FastAPI (Python)
*   **Database & Auth:** Firebase, Firestore
*   **Bundler:** Webpack

## Getting Started

### Prerequisites

*   Node.js and npm
*   Python 3.8+ and pip
*   A Firebase project

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install frontend dependencies:**
    ```bash
    npm install
    ```

3.  **Install backend dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

1.  **Configure Firebase:** You will need to add your Firebase project configuration to the application.
2.  **Start the backend server:**
    ```bash
    uvicorn backend.main:app --reload
    ```
3.  **Start the frontend development server:**
    ```bash
    npm start
    ```

The application will be available at `http://localhost:8080`.

## Project Structure

The repository is organized into several key directories:

*   `src/`: The main source code for the React frontend.
*   `backend/`: The source code for the FastAPI backend.
*   `docs/`: Detailed documentation for the project.
*   `public/`: Static assets for the frontend.

## Documentation

This project includes comprehensive documentation to help you understand and use the application:

*   **[Owner's Guide](docs/OWNERS_GUIDE.md):** A technical overview for owners and administrators.
*   **[User's Guide](docs/USERS_GUIDE.md):** A guide for end-users of the application.
*   **[Admin Guide](docs/ADMIN_GUIDE.md):** A guide for administrators.
*   **[Investor's Briefing](docs/INVESTORS_BRIEFING.md):** A high-level overview for stakeholders.
*   **[Recruiter's Overview](docs/RECRUITERS_OVERVIEW.md):** A technical overview for recruiters.

## Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request.

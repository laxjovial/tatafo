import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import LoginPage from './LoginPage';

describe('LoginPage', () => {
    test('renders login form', () => {
        render(<LoginPage />);

        // Check for the main title
        expect(screen.getByText('Login')).toBeInTheDocument();

        // Check for email and password input fields
        expect(screen.getByLabelText('Email')).toBeInTheDocument();
        expect(screen.getByLabelText('Password')).toBeInTheDocument();

        // Check for the login button
        expect(screen.getByRole('button', { name: 'Login' })).toBeInTheDocument();
    });

    test('displays error message on login failure', () => {
        // This test would require mocking the fetch API and simulating a failed login attempt
    });

    test('calls onLoginSuccess on successful login', () => {
        // This test would require mocking the fetch API and simulating a successful login
    });
});

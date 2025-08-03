import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import RegisterPage from './RegisterPage';

describe('RegisterPage', () => {
    test('renders registration form', () => {
        render(<RegisterPage />);

        // Check for the main title
        expect(screen.getByText('Register')).toBeInTheDocument();

        // Check for username, email, and password input fields
        expect(screen.getByLabelText('Username')).toBeInTheDocument();
        expect(screen.getByLabelText('Email')).toBeInTheDocument();
        expect(screen.getByLabelText('Password')).toBeInTheDocument();

        // Check for the register button
        expect(screen.getByRole('button', { name: 'Register' })).toBeInTheDocument();
    });

    test('displays error message on registration failure', () => {
        // This test would require mocking the fetch API and simulating a failed registration attempt
    });

    test('displays success message on successful registration', () => {
        // This test would require mocking the fetch API and simulating a successful registration
    });
});

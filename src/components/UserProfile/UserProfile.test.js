import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import UserProfile from './UserProfile';

// Mock the useAuth hook
jest.mock('../../hooks/useAuth', () => ({
    __esModule: true,
    default: () => ({
        user: { uid: 'test-user-id' },
        auth: {
            currentUser: {
                getIdToken: () => Promise.resolve('test-token'),
            },
        },
    }),
}));

describe('UserProfile', () => {
    test('renders user profile page', async () => {
        // Mock the fetch API to return some user data
        global.fetch = jest.fn(() =>
            Promise.resolve({
                ok: true,
                json: () => Promise.resolve({
                    username: 'testuser',
                    email: 'test@example.com',
                    phone: '123-456-7890',
                    address: '123 Main St',
                    bio: 'This is a test bio',
                    tier: 'free',
                }),
            })
        );

        render(<UserProfile userId="test-user-id" />);

        // Check for the main title
        expect(screen.getByText('User Profile')).toBeInTheDocument();

        // Check that user data is displayed
        // We use findBy because the data is fetched asynchronously
        expect(await screen.findByDisplayValue('testuser')).toBeInTheDocument();
        expect(await screen.findByDisplayValue('test@example.com')).toBeInTheDocument();
    });
});

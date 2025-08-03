import React, { useState } from 'react';
// import { logEvent } from '../../services/analytics'; // This will be un-commented later
import './RegisterPage.css';

import { FASTAPI_BASE_URL } from '../../config';

const RegisterPage = ({ onRegisterSuccess, onNavigateToLogin, auth }) => {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [username, setUsername] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [message, setMessage] = useState('');

    const handleRegister = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError('');
        setMessage('');
        try {
            if (!auth) {
                throw new Error("Firebase Auth is not initialized. Please refresh the page.");
            }

            const response = await fetch(`${FASTAPI_BASE_URL}/register`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, password, username }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Registration failed.');
            }

            const data = await response.json();
            setMessage(data.message || 'Registration successful! Please log in.');
            // await logEvent('user_registration', { email: email, username: username, uid: data.uid }, true);
            onRegisterSuccess();
        } catch (err) {
            console.error("Registration error:", err);
            setError(err.message || 'An unexpected error occurred during registration.');
            // await logEvent('user_registration', { email: email, username: username, error: err.message }, false, err.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="register-page">
            <div className="register-container">
                <h2 className="register-title">Register</h2>
                {error && <div className="error-message">{error}</div>}
                {message && <div className="success-message">{message}</div>}
                <form onSubmit={handleRegister} className="register-form">
                    <div>
                        <label className="form-label" htmlFor="username">Username</label>
                        <input
                            type="text"
                            id="username"
                            value={username}
                            onChange={(e) => setUsername(e.target.value)}
                            className="form-input"
                            placeholder="Your Username"
                            required
                        />
                    </div>
                    <div>
                        <label className="form-label" htmlFor="email">Email</label>
                        <input
                            type="email"
                            id="email"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            className="form-input"
                            placeholder="your@example.com"
                            required
                        />
                    </div>
                    <div>
                        <label className="form-label" htmlFor="password">Password</label>
                        <input
                            type="password"
                            id="password"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            className="form-input"
                            placeholder="********"
                            required
                            minLength="6"
                        />
                    </div>
                    <button
                        type="submit"
                        className="submit-button"
                        disabled={loading}
                    >
                        {loading ? (
                            <svg className="spinner" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                            </svg>
                        ) : 'Register'}
                    </button>
                </form>
                <div className="navigation-link">
                    <p>Already have an account?</p>
                    <button
                        onClick={onNavigateToLogin}
                        className="link-button"
                    >
                        Login here
                    </button>
                </div>
            </div>
        </div>
    );
};

export default RegisterPage;

import React, { useState } from 'react';
import { signInWithCustomToken } from 'firebase/auth';
// import { logEvent } from '../../services/analytics'; // This will be un-commented later
import './LoginPage.css';

import { FASTAPI_BASE_URL } from '../../config';

const LoginPage = ({ onLoginSuccess, onNavigateToRegister, auth }) => {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    const handleLogin = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError('');
        try {
            if (!auth) {
                throw new Error("Firebase Auth is not initialized. Please refresh the page.");
            }

            const response = await fetch(`${FASTAPI_BASE_URL}/login`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, password }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Login failed. Please check your credentials.');
            }

            const data = await response.json();
            const customToken = data.custom_token;
            const uid = data.uid;

            await signInWithCustomToken(auth, customToken);

            console.log("Logged in successfully with Firebase Client SDK:", uid);
            // await logEvent('user_login', { email: email, uid: uid }, true);
            onLoginSuccess(uid);

        } catch (err) {
            console.error("Login error:", err);
            setError(err.message || 'An unexpected error occurred during login.');
            // await logEvent('user_login', { email: email, error: err.message }, false, err.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="login-page">
            <div className="login-container">
                <h2 className="login-title">Login</h2>
                {error && <div className="error-message">{error}</div>}
                <form onSubmit={handleLogin} className="login-form">
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
                        ) : 'Login'}
                    </button>
                </form>
                <div className="navigation-link">
                    <p>Don't have an account?</p>
                    <button
                        onClick={onNavigateToRegister}
                        className="link-button"
                    >
                        Register here
                    </button>
                </div>
            </div>
        </div>
    );
};

export default LoginPage;

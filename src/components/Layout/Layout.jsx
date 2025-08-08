import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import useAuth from '../../hooks/useAuth';
import './Layout.css';

const Layout = ({ children }) => {
    const { isAuthenticated, logout, isAdmin } = useAuth();
    const navigate = useNavigate();

    const handleLogout = async () => {
        await logout();
        navigate('/login');
    };

    return (
        <div className="layout">
            <header className="header">
                <div className="header-container">
                    <Link to="/" className="logo">Intelli-Agent</Link>
                    <nav className="nav">
                        {isAuthenticated && (
                            <ul>
                                <li><Link to="/dashboard">Dashboard</Link></li>
                                <li><Link to="/assistant">Assistant</Link></li>
                                <li><Link to="/profile">Profile</Link></li>
                                {isAdmin && <li><Link to="/admin">Admin</Link></li>}
                                <li><a href="/docs/USERS_GUIDE.md" target="_blank" rel="noopener noreferrer">User Guide</a></li>
                                <li><button onClick={handleLogout} className="logout-button">Logout</button></li>
                            </ul>
                        )}
                    </nav>
                </div>
            </header>
            <main className="main-content">
                {children}
            </main>
            <footer className="footer">
                <p>&copy; 2024 Intelli-Agent. All rights reserved. This is a disclaimer text.</p>
            </footer>
        </div>
    );
};

export default Layout;

import React from 'react';
import { BrowserRouter as Router, Route, Routes, Navigate } from 'react-router-dom';
import useAuth from './hooks/useAuth';
import LoginPage from './components/LoginPage';
import RegisterPage from './components/RegisterPage';
import DashboardPage from './components/DashboardPage';
import AssistantPage from './components/AssistantPage';
import UpgradePage from './components/UpgradePage';
import UserProfilePage from './components/UserProfilePage';


const App = () => {
    const auth = useAuth();

    return (
        <Router>
            <Routes>
                <Route path="/login" element={
                    !auth.isAuthenticated ? (
                        <LoginPage
                            onLoginSuccess={(uid) => auth.login(uid)}
                            onNavigateToRegister={() => <Navigate to="/register" />}
                            auth={auth}
                        />
                    ) : (
                        <Navigate to="/dashboard" />
                    )
                } />
                <Route path="/register" element={
                    !auth.isAuthenticated ? (
                        <RegisterPage
                            onRegisterSuccess={() => <Navigate to="/login" />}
                            onNavigateToLogin={() => <Navigate to="/login" />}
                            auth={auth}
                        />
                    ) : (
                        <Navigate to="/dashboard" />
                    )
                } />
                <Route path="/dashboard" element={
                    auth.isAuthenticated ? (
                        <DashboardPage auth={auth} />
                    ) : (
                        <Navigate to="/login" />
                    )
                } />
                <Route path="/assistant" element={
                    auth.isAuthenticated ? (
                        <AssistantPage auth={auth} />
                    ) : (
                        <Navigate to="/login" />
                    )
                } />
                <Route path="/upgrade" element={
                    auth.isAuthenticated ? (
                        <UpgradePage />
                    ) : (
                        <Navigate to="/login" />
                    )
                } />
                <Route path="/profile" element={
                    auth.isAuthenticated ? (
                        <UserProfilePage auth={auth} />
                    ) : (
                        <Navigate to="/login" />
                    )
                } />
                <Route path="/" element={<Navigate to={auth.isAuthenticated ? "/dashboard" : "/login"} />} />
            </Routes>
        </Router>
    );
};

export default App;

import React from 'react';
import { BrowserRouter as Router, Route, Routes, Navigate } from 'react-router-dom';
import useAuth from './hooks/useAuth';
import Layout from './components/Layout/Layout';
import PrivateRoute from './components/PrivateRoute/PrivateRoute';

// Page Imports
import LoginPage from './pages/LoginPage';
import RegisterPage from './pages/RegisterPage';
import DashboardPage from './pages/DashboardPage';
import AIAssistantPage from './pages/AIAssistantPage';
import SubscriptionPage from './pages/SubscriptionPage';
import UserProfilePage from './pages/UserProfilePage';
import AdminDashboardPage from './pages/AdminDashboardPage';
import ForgotPasswordPage from './pages/ForgotPasswordPage';
import ChangePasswordPage from './pages/ChangePasswordPage';
import ResetPasswordPage from './pages/ResetPasswordPage';
import IntegrationsPage from './pages/IntegrationsPage';


const App = () => {
    const { isAuthenticated, loading } = useAuth();

    if (loading) {
        return <div>Loading...</div>; // Or a more sophisticated loading spinner
    }

    return (
        <Router>
            <Layout>
                <Routes>
                    {/* Public Routes */}
                    <Route path="/login" element={!isAuthenticated ? <LoginPage /> : <Navigate to="/dashboard" />} />
                    <Route path="/register" element={!isAuthenticated ? <RegisterPage /> : <Navigate to="/dashboard" />} />
                    <Route path="/forgot-password" element={!isAuthenticated ? <ForgotPasswordPage /> : <Navigate to="/dashboard" />} />
                    <Route path="/reset-password" element={!isAuthenticated ? <ResetPasswordPage /> : <Navigate to="/dashboard" />} />


                    {/* Private Routes */}
                    <Route path="/dashboard" element={<PrivateRoute><DashboardPage /></PrivateRoute>} />
                    <Route path="/assistant" element={<PrivateRoute><AIAssistantPage /></PrivateRoute>} />
                    <Route path="/subscription" element={<PrivateRoute><SubscriptionPage /></PrivateRoute>} />
                    <Route path="/profile" element={<PrivateRoute><UserProfilePage /></PrivateRoute>} />
                    <Route path="/change-password" element={<PrivateRoute><ChangePasswordPage /></PrivateRoute>} />
                    <Route path="/integrations" element={<PrivateRoute><IntegrationsPage /></PrivateRoute>} />

                    {/* Admin Route */}
                    <Route path="/admin" element={<PrivateRoute adminOnly={true}><AdminDashboardPage /></PrivateRoute>} />

                    {/* Redirect root path */}
                    <Route path="/" element={<Navigate to={isAuthenticated ? "/dashboard" : "/login"} />} />

                    {/* Fallback for any other path */}
                    <Route path="*" element={<div>Page Not Found</div>} />
                </Routes>
            </Layout>
        </Router>
    );
};

export default App;

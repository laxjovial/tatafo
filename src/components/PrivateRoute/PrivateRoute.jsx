import React from 'react';
import { Navigate } from 'react-router-dom';
import useAuth from '../../hooks/useAuth';

const PrivateRoute = ({ children, adminOnly = false }) => {
    const { isAuthenticated, isAdmin, loading } = useAuth();

    if (loading) {
        // You can return a loading spinner here
        return <div>Loading...</div>;
    }

    if (!isAuthenticated) {
        // Redirect them to the /login page, but save the current location they were
        // trying to go to. This allows us to send them along to that page after they login.
        return <Navigate to="/login" replace />;
    }

    if (adminOnly && !isAdmin) {
        // Redirect them to a generic dashboard or a "not authorized" page
        return <Navigate to="/dashboard" replace />;
    }

    return children;
};

export default PrivateRoute;

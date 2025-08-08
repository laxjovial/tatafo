
import React, { useState } from 'react';
import useAuth from '../hooks/useAuth';
import { FASTAPI_BASE_URL } from '../config';
import './IntegrationsPage.css';

const IntegrationsPage = () => {
    const { auth } = useAuth();
    const [query, setQuery] = useState('');
    const [results, setResults] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleConnect = async (provider) => {
        setLoading(true);
        setError(null);
        try {
            const idToken = await auth.currentUser.getIdToken(true);
            const response = await fetch(`${FASTAPI_BASE_URL}/integrations/${provider}/connect/start`, {
                headers: {
                    'Authorization': `Bearer ${idToken}`,
                },
            });

            if (!response.ok) {
                throw new Error(`Failed to start connection process for ${provider}.`);
            }

            const data = await response.json();
            window.location.href = data.authorization_url;
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const handleQuery = async (provider) => {
        setLoading(true);
        setError(null);
        try {
            const idToken = await auth.currentUser.getIdToken(true);
            const response = await fetch(`${FASTAPI_BASE_URL}/integrations/${provider}/query?query=${query}`, {
                headers: {
                    'Authorization': `Bearer ${idToken}`,
                },
            });

            if (!response.ok) {
                throw new Error(`Failed to query ${provider}.`);
            }
            const data = await response.json();
            setResults(data);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="integrations-page">
            <h2>Third-Party Integrations</h2>
            <div className="integrations-grid">
                <div className="integration-card">
                    <h3>Google Drive</h3>
                    <button onClick={() => handleConnect('google-drive')} disabled={loading}>
                        Connect
                    </button>
                </div>
                <div className="integration-card">
                    <h3>OneDrive</h3>
                    <button onClick={() => handleConnect('one-drive')} disabled={loading}>
                        Connect
                    </button>
                </div>
            </div>

            <div className="query-section">
                <h3>Query Integrations</h3>
                <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="Search your connected drives..."
                    className="query-input"
                />
                <div className="query-buttons">
                    <button onClick={() => handleQuery('google-drive')} disabled={loading}>
                        Query Google Drive
                    </button>
                    <button onClick={() => handleQuery('one-drive')} disabled={loading}>
                        Query OneDrive
                    </button>
                </div>
            </div>

            {error && <div className="error-message">{error}</div>}

            <div className="results-section">
                <h3>Results</h3>
                {loading ? (
                    <div>Loading...</div>
                ) : (
                    <ul>
                        {results.map((result, index) => (
                            <li key={index}>
                                {result.name} ({result.type}) - from {result.source}
                            </li>
                        ))}
                    </ul>
                )}
            </div>

            <div className="database-section">
                <h3>Connect to a Database</h3>
                <input
                    type="text"
                    placeholder="Enter your database connection string..."
                    className="query-input"
                />
                <button disabled={loading}>
                    Connect Database
                </button>
            </div>
        </div>
    );

};

export default IntegrationsPage;

import React, { useState, useEffect } from 'react';

const UserProfilePage = ({ auth }) => {
    const [apiKeys, setApiKeys] = useState([]);
    const [newApiKey, setNewApiKey] = useState({ provider: '', key: '' });
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    useEffect(() => {
        const fetchApiKeys = async () => {
            setLoading(true);
            try {
                const response = await fetch('/api/user/api-keys', {
                    headers: { 'Authorization': `Bearer ${auth.token}` },
                });
                if (!response.ok) throw new Error('Failed to fetch API keys.');
                const data = await response.json();
                setApiKeys(data);
            } catch (err) {
                setError(err.message);
            } finally {
                setLoading(false);
            }
        };
        fetchApiKeys();
    }, [auth.token]);

    const handleAddApiKey = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError('');
        try {
            const response = await fetch('/api/user/api-keys', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${auth.token}`,
                },
                body: JSON.stringify(newApiKey),
            });
            if (!response.ok) throw new Error('Failed to add API key.');
            const data = await response.json();
            setApiKeys([...apiKeys, data]);
            setNewApiKey({ provider: '', key: '' });
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div>
            <h2>User Profile</h2>
            <h3>Your API Keys</h3>
            {loading && <p>Loading...</p>}
            {error && <p>Error: {error}</p>}
            <ul>
                {apiKeys.map((apiKey) => (
                    <li key={apiKey.provider}>{apiKey.provider}</li>
                ))}
            </ul>
            <form onSubmit={handleAddApiKey}>
                <input
                    type="text"
                    placeholder="Provider"
                    value={newApiKey.provider}
                    onChange={(e) => setNewApiKey({ ...newApiKey, provider: e.target.value })}
                    required
                />
                <input
                    type="password"
                    placeholder="API Key"
                    value={newApiKey.key}
                    onChange={(e) => setNewApiKey({ ...newApiKey, key: e.target.value })}
                    required
                />
                <button type="submit" disabled={loading}>Add API Key</button>
            </form>
        </div>
    );
};

export default UserProfilePage;

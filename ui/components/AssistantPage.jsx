import React, { useState } from 'react';

const AssistantPage = ({ auth }) => {
    const [query, setQuery] = useState('');
    const [messages, setMessages] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!query.trim()) return;

        const newMessages = [...messages, { sender: 'user', text: query }];
        setMessages(newMessages);
        setQuery('');
        setLoading(true);
        setError('');

        try {
            const response = await fetch('/api/assistant', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${auth.token}`,
                },
                body: JSON.stringify({ query }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                setError(errorData.detail || 'Failed to get response from assistant.');
                return;
            }

            const data = await response.json();
            setMessages([...newMessages, { sender: 'assistant', text: data.response }]);

        } catch (err) {
            setError(err.message || 'An unexpected error occurred.');
        } finally {
            setLoading(false);
        }
    };

    if (auth.user.tier === 'free') {
        return (
            <div>
                <h2>AI Assistant</h2>
                <p>The AI assistant is not available for free tier users.</p>
                <a href="/upgrade">Upgrade to a paid tier to use this feature.</a>
            </div>
        );
    }

    return (
        <div>
            <h2>AI Assistant</h2>
            <div style={{ height: '400px', overflowY: 'scroll', border: '1px solid #ccc', padding: '10px' }}>
                {messages.map((msg, index) => (
                    <div key={index} style={{ textAlign: msg.sender === 'user' ? 'right' : 'left' }}>
                        <p><strong>{msg.sender}:</strong> {msg.text}</p>
                    </div>
                ))}
            </div>
            <form onSubmit={handleSubmit}>
                <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="Ask the assistant anything..."
                    disabled={loading}
                />
                <button type="submit" disabled={loading}>
                    {loading ? 'Sending...' : 'Send'}
                </button>
            </form>
            {error && <div>Error: {error}</div>}
        </div>
    );
};

export default AssistantPage;

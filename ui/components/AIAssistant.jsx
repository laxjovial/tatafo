import React, { useState, useEffect, useRef } from 'react';

const AIAssistant = ({ auth }) => {
    const [message, setMessage] = useState('');
    const [chatHistory, setChatHistory] = useState([]);
    const [loading, setLoading] = useState(false);
    const chatEndRef = useRef(null);

    const scrollToBottom = () => {
        chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [chatHistory]);

    const handleSendMessage = async (e) => {
        e.preventDefault();
        if (!message.trim()) return;

        const newChatHistory = [...chatHistory, { role: 'user', content: message }];
        setChatHistory(newChatHistory);
        setMessage('');
        setLoading(true);

        try {
            const idToken = await auth.currentUser.getIdToken(true);
            const response = await fetch('/tools/chat/agent', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${idToken}`
                },
                body: JSON.stringify({
                    prompt: message,
                    chat_history: newChatHistory,
                    user_token: idToken
                }),
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || 'Failed to get response from AI assistant.');
            }

            setChatHistory([...newChatHistory, { role: 'assistant', content: data.response }]);
        } catch (error) {
            setChatHistory([...newChatHistory, { role: 'assistant', content: `Error: ${error.message}` }]);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div>
            <h2>AI Assistant</h2>
            <div>
                {chatHistory.map((chat, index) => (
                    <div key={index} className={chat.role}>
                        <p><strong>{chat.role}:</strong> {chat.content}</p>
                    </div>
                ))}
                <div ref={chatEndRef} />
            </div>
            <form onSubmit={handleSendMessage}>
                <input
                    type="text"
                    value={message}
                    onChange={(e) => setMessage(e.target.value)}
                    placeholder="Ask the AI assistant..."
                    disabled={loading}
                />
                <button type="submit" disabled={loading}>
                    {loading ? 'Sending...' : 'Send'}
                </button>
            </form>
        </div>
    );
};

export default AIAssistant;

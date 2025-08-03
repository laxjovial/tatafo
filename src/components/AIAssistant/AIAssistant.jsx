import React, { useState, useEffect, useRef } from 'react';
import useAuth from '../../hooks/useAuth';
import { FASTAPI_BASE_URL } from '../../config';
import './AIAssistant.css';

const AIAssistant = () => {
    const { auth } = useAuth();
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
            const response = await fetch(`${FASTAPI_BASE_URL}/tools/chat/agent`, {
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

    const handleFileUpload = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        setLoading(true);
        const formData = new FormData();
        formData.append('file', file);
        formData.append('section', 'general'); // Or make this dynamic

        try {
            const idToken = await auth.currentUser.getIdToken(true);
            const response = await fetch(`${FASTAPI_BASE_URL}/docs/upload`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${idToken}`,
                },
                body: formData,
            });

            const data = await response.json();

            if (!response.ok) {
                if (response.status === 413) {
                    throw new Error("Upload failed: The file is too large and would exceed your storage limit.");
                }
                throw new Error(data.detail || 'Failed to upload file.');
            }

            setChatHistory([...chatHistory, { role: 'assistant', content: `File uploaded successfully: ${file.name}` }]);
        } catch (error) {
            setChatHistory([...chatHistory, { role: 'assistant', content: `Error: ${error.message}` }]);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="ai-assistant-container">
            <div className="chat-history">
                {chatHistory.map((chat, index) => (
                    <div key={index} className={`chat-message ${chat.role}`}>
                        <p><strong>{chat.role}:</strong> {chat.content}</p>
                    </div>
                ))}
                <div ref={chatEndRef} />
            </div>
            <div className="chat-input-area">
                <form onSubmit={handleSendMessage} className="message-form">
                    <input
                        type="text"
                        value={message}
                        onChange={(e) => setMessage(e.target.value)}
                        placeholder="Ask the AI assistant..."
                        disabled={loading}
                        className="message-input"
                    />
                    <button type="submit" disabled={loading} className="send-button">
                        {loading ? 'Sending...' : 'Send'}
                    </button>
                </form>
                <div className="file-upload">
                    <label htmlFor="file-upload" className="file-upload-label">
                        Upload File
                    </label>
                    <input
                        id="file-upload"
                        type="file"
                        onChange={handleFileUpload}
                        disabled={loading}
                        className="file-upload-input"
                    />
                </div>
            </div>
        </div>
    );
};

export default AIAssistant;

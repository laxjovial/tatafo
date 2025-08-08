import React, { useState, useEffect, useRef } from 'react';
import useAuth from '../../hooks/useAuth';
import { FASTAPI_BASE_URL } from '../../config';
import './AIAssistant.css';

const AIAssistant = () => {
    const { auth } = useAuth();
    const [message, setMessage] = useState('');
    const [chatHistory, setChatHistory] = useState([]);

    const [sessions, setSessions] = useState([]);
    const [currentSessionId, setCurrentSessionId] = useState(null);

    const [loading, setLoading] = useState(false);
    const chatEndRef = useRef(null);

    const scrollToBottom = () => {
        chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [chatHistory]);


    useEffect(() => {
        const fetchSessions = async () => {
            if (!auth || !auth.currentUser) return;
            setLoading(true);
            try {
                const idToken = await auth.currentUser.getIdToken(true);
                const response = await fetch(`${FASTAPI_BASE_URL}/chat/sessions`, {
                    headers: { 'Authorization': `Bearer ${idToken}` },
                });
                if (response.ok) {
                    const data = await response.json();
                    setSessions(data);
                }
            } catch (error) {
                console.error("Failed to fetch sessions:", error);
            } finally {
                setLoading(false);
            }
        };
        fetchSessions();
    }, [auth]);

    const handleNewSession = async () => {
        setLoading(true);
        try {
            const idToken = await auth.currentUser.getIdToken(true);
            const response = await fetch(`${FASTAPI_BASE_URL}/chat/sessions`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${idToken}`
                },
                body: JSON.stringify({ title: "New Chat" })
            });
            if (response.ok) {
                const data = await response.json();
                setCurrentSessionId(data.session_id);
                setChatHistory([]);
                fetchSessions(); // Refresh the session list
            }
        } catch (error) {
            console.error("Failed to create new session:", error);
        } finally {
            setLoading(false);
        }
    };

    const handleSelectSession = async (sessionId) => {
        setLoading(true);
        try {
            const idToken = await auth.currentUser.getIdToken(true);
            const response = await fetch(`${FASTAPI_BASE_URL}/chat/sessions/${sessionId}`, {
                headers: { 'Authorization': `Bearer ${idToken}` }
            });
            if (response.ok) {
                const data = await response.json();
                setCurrentSessionId(sessionId);
                setChatHistory(data);
            }
        } catch (error) {
            console.error("Failed to fetch session messages:", error);
        } finally {
            setLoading(false);
        }
    };

    const handleSendMessage = async (e) => {
        e.preventDefault();
        if (!message.trim() || !currentSessionId) return;


        const newChatHistory = [...chatHistory, { role: 'user', content: message }];
        setChatHistory(newChatHistory);
        setMessage('');
        setLoading(true);

        try {
            const idToken = await auth.currentUser.getIdToken(true);

            // First, save the user's message
            await fetch(`${FASTAPI_BASE_URL}/chat/sessions/${currentSessionId}/messages`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${idToken}`
                },
                body: JSON.stringify({ role: 'user', content: message })
            });

            // Then, get the AI's response

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


            const aiResponse = data.response;
            setChatHistory([...newChatHistory, { role: 'assistant', content: aiResponse }]);

            // Save the AI's message
            await fetch(`${FASTAPI_BASE_URL}/chat/sessions/${currentSessionId}/messages`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${idToken}`
                },
                body: JSON.stringify({ role: 'assistant', content: aiResponse })
            });


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

    const [systemPrompt, setSystemPrompt] = useState('You are a helpful assistant.');
    const [llmProvider, setLlmProvider] = useState('gemini');

    const handleSystemPromptChange = (e) => {
        setSystemPrompt(e.target.value);
    };

    const handleLlmProviderChange = (e) => {
        setLlmProvider(e.target.value);
    };

    return (

        <div className="ai-assistant-page">
            <div className="sessions-sidebar">
                <h3>Chat Sessions</h3>
                <button onClick={handleNewSession}>+ New Chat</button>
                <ul>
                    {sessions.map(session => (
                        <li key={session.id} onClick={() => handleSelectSession(session.id)}>
                            {session.title}
                        </li>
                    ))}
                </ul>
            </div>
            <div className="ai-assistant-container">
                <div className="ai-assistant-settings">
                    <textarea
                        value={systemPrompt}
                        onChange={handleSystemPromptChange}
                        placeholder="Enter a system prompt..."
                        className="system-prompt-input"
                    />
                    <select value={llmProvider} onChange={handleLlmProviderChange} className="llm-provider-select">
                        <option value="gemini">Gemini</option>
                        <option value="togetherai">Together AI</option>
                    </select>
                </div>
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

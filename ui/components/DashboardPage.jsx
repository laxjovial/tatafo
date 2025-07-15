import React, { useState } from 'react';
import { Link } from 'react-router-dom';

const DashboardPage = ({ auth }) => {
    const [tool, setTool] = useState('');
    const [params, setParams] = useState({});
    const [result, setResult] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    const handleToolChange = (e) => {
        setTool(e.target.value);
        setParams({});
        setResult('');
    };

    const handleParamChange = (e) => {
        setParams({ ...params, [e.target.name]: e.target.value });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError('');
        setResult('');

        try {
            const response = await fetch(`/api/run-tool/${tool}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${auth.token}`,
                },
                body: JSON.stringify(params),
            });

            if (!response.ok) {
                const errorData = await response.json();

                setError(errorData.detail || 'Tool execution failed.');
                return;


            const data = await response.json();
            setResult(JSON.stringify(data, null, 2));

        } catch (err) {
            setError(err.message || 'An unexpected error occurred.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div>
            <h2>Dashboard</h2>
            <p>Welcome, {auth.user.username}!</p>
            <p>Tier: {auth.user.tier}</p>

            <Link to="/assistant">Go to AI Assistant</Link>
            <Link to="/profile">Go to Profile</Link>


            <form onSubmit={handleSubmit}>
                <div>
                    <label htmlFor="tool">Select Tool</label>
                    <select id="tool" value={tool} onChange={handleToolChange}>
                        <option value="">--Select a tool--</option>
                        <option value="finance_get_historical_stock_prices">Get Historical Stock Prices</option>
                        <option value="crypto_get_historical_crypto_price">Get Historical Crypto Prices</option>
                        {/* Add other tools as they become available */}
                    </select>
                </div>

                {tool === 'finance_get_historical_stock_prices' && (
                    <div>
                        <label htmlFor="ticker">Ticker</label>
                        <input type="text" name="ticker" onChange={handleParamChange} required />
                        <label htmlFor="provider">Provider</label>
                        <select name="provider" onChange={handleParamChange}>
                            <option value="alphavantage">Alpha Vantage</option>
                            {/* Add other finance providers here */}
                        </select>
                    </div>
                )}

                {tool === 'crypto_get_historical_crypto_price' && (
                    <div>
                        <label htmlFor="symbol">Crypto Symbol</label>
                        <input type="text" name="symbol" onChange={handleParamChange} required />
                        <label htmlFor="provider">Provider</label>
                        <select name="provider" onChange={handleParamChange}>
                            <option value="coingecko">CoinGecko</option>
                            {/* Add other crypto providers here */}
                        </select>
                    </div>
                )}

                <button type="submit" disabled={!tool || loading || (tool === 'finance_get_historical_stock_prices' && auth.user.tier === 'free') || (tool === 'crypto_get_historical_crypto_price' && auth.user.tier === 'free')}>
                    {loading ? 'Running...' : 'Run Tool'}
                </button>
                {(tool === 'finance_get_historical_stock_prices' && auth.user.tier === 'free') && <Link to="/upgrade">Upgrade to a paid tier to use this feature.</Link>}
                {(tool === 'crypto_get_historical_crypto_price' && auth.user.tier === 'free') && <Link to="/upgrade">Upgrade to a paid tier to use this feature.</Link>}

            </form>

            {error && <div>Error: {error}</div>}
            {result && (
                <div>
                    <h3>Result:</h3>
                    <pre>{result}</pre>
                </div>
            )}
        </div>
    );
};

export default DashboardPage;

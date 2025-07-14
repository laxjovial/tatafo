import React from 'react';

const UpgradePage = () => {
    const handleUpgrade = () => {
        // In a real application, this would redirect the user to a payment page.
        alert('Redirecting to payment page...');
    };

    return (
        <div>
            <h2>Upgrade Your Plan</h2>
            <p>You are currently on the free tier. Upgrade to a paid tier to access more features.</p>
            <div>
                <h3>Paid Tier</h3>
                <p>Includes access to all tools.</p>
                <p>Price: $10/month</p>
                <button onClick={handleUpgrade}>Upgrade to Paid Tier</button>
            </div>
            <div>
                <h3>Premium Tier</h3>
                <p>Includes access to all tools and the AI assistant.</p>
                <p>Price: $20/month</p>
                <button onClick={handleUpgrade}>Upgrade to Premium Tier</button>
            </div>
        </div>
    );
};

export default UpgradePage;

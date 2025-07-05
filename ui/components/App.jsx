const App = () => {
    const [currentPage, setCurrentPage] = useState('home');
    const [currentUserId, setCurrentUserId] = useState(null);
    const [isAnalyticsInitialized, setIsAnalyticsInitialized] = useState(false);

    useEffect(() => {
        if (!auth) {
            console.warn("Firebase Auth not available. Cannot initialize analytics with user context.");
            setIsAnalyticsInitialized(true);
            return;
        }

        const unsubscribe = onAuthStateChanged(auth, async (user) => {
            let userIdToUse = null;
            if (user) {
                userIdToUse = user.uid;
            } else {
                try {
                    if (typeof __initial_auth_token !== 'undefined' && __initial_auth_token) {
                        await signInWithCustomToken(auth, __initial_auth_token);
                    } else {
                        await signInAnonymously(auth);
                    }
                    userIdToUse = auth.currentUser?.uid || crypto.randomUUID();
                } catch (anonError) {
                    console.error("Error signing in anonymously for analytics:", anonError);
                    userIdToUse = 'anonymous_error';
                }
            }
            setCurrentUserId(userIdToUse);

            if (db && userIdToUse) {
                initializeAnalytics(db, auth, appId, userIdToUse);
                setIsAnalyticsInitialized(true);
            } else {
                console.warn("Firestore or User ID not available for analytics initialization.");
                setIsAnalyticsInitialized(true);
            }
        });

        return () => unsubscribe();
    }, []);

    useEffect(() => {
        if (isAnalyticsInitialized && currentUserId) {
            (async () => {
                try {
                    await logEvent('page_view', { page_name: currentPage, user_id: currentUserId });
                    console.log(`Analytics: Logged page_view for '${currentPage}' by user '${currentUserId}'`);
                } catch (err) {
                    console.error("Failed to log analytics event:", err);
                }
            })();
        }
    }, [currentPage, isAnalyticsInitialized, currentUserId]);

    const renderPage = () => {
        switch (currentPage) {
            case 'home':
                return (
                    <div className="text-center p-10">
                        <h2 className="text-3xl font-bold text-gray-800 mb-4">Welcome to the AI Assistant!</h2>
                        <p className="text-lg text-gray-600">Explore the features using the navigation below.</p>
                    </div>
                );
            case 'analytics':
                return <AnalyticsDashboard />;
            case 'profile':
                return <UserProfile userId={currentUserId} />;
            default:
                return (
                    <div className="text-center p-10">
                        <h2 className="text-3xl font-bold text-gray-800 mb-4">Page Not Found</h2>
                        <p className="text-lg text-gray-600">The page you are looking for does not exist.</p>
                    </div>
                );
        }
    };

    return (
        <div className="min-h-screen bg-gray-50">
            <nav className="bg-indigo-700 p-4 shadow-md">
                <ul className="flex justify-center space-x-6">
                    <li>
                        <button
                            onClick={() => setCurrentPage('home')}
                            className="text-white hover:text-indigo-200 text-lg font-medium transition duration-300"
                        >
                            Home
                        </button>
                    </li>
                    <li>
                        <button
                            onClick={() => setCurrentPage('analytics')}
                            className="text-white hover:text-indigo-200 text-lg font-medium transition duration-300"
                        >
                            Analytics
                        </button>
                    </li>
                    <li>
                        <button
                            onClick={() => setCurrentPage('profile')}
                            className="text-white hover:text-indigo-200 text-lg font-medium transition duration-300"
                        >
                            User Profile
                        </button>
                    </li>
                </ul>
            </nav>
            <main className="container mx-auto p-4">
                {isAnalyticsInitialized ? renderPage() : (
                    <div className="text-center py-20 text-xl text-indigo-600">Initializing application...</div>
                )}
            </main>
        </div>
    );
};

export default App;

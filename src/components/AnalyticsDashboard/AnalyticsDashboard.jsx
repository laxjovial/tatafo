import React, { useState, useEffect, useCallback } from 'react';
import { collection, query, where, getDocs } from 'firebase/firestore';
// import { logEvent } from '../../services/analytics';
import './AnalyticsDashboard.css';

const AnalyticsDashboard = ({ db, auth, appId, currentUserId }) => {
    const [events, setEvents] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [filterEventType, setFilterEventType] = useState('');
    const [filterUserId, setFilterUserId] = useState('');
    const [filterStartDate, setFilterStartDate] = useState('');
    const [filterEndDate, setFilterEndDate] = useState('');

    const fetchAnalyticsEvents = useCallback(async () => {
        if (!db || !appId || !currentUserId) {
            setError("Firebase or User ID not available for analytics. Please ensure you are logged in.");
            setLoading(false);
            return;
        }

        setLoading(true);
        setError(null);
        try {
            const analyticsCollectionRef = collection(db, `artifacts/${appId}/public/data/analytics_events`);
            let q = query(analyticsCollectionRef);

            if (filterEventType) {
                q = query(q, where('event_type', '==', filterEventType));
            }
            if (filterUserId) {
                q = query(q, where('user_id', '==', filterUserId));
            }

            const querySnapshot = await getDocs(q);
            let fetchedEvents = querySnapshot.docs.map(doc => ({ id: doc.id, ...doc.data() }));

            if (filterStartDate) {
                const start = new Date(filterStartDate).toISOString();
                fetchedEvents = fetchedEvents.filter(event => event.timestamp >= start);
            }
            if (filterEndDate) {
                const end = new Date(filterEndDate).toISOString();
                fetchedEvents = fetchedEvents.filter(event => event.timestamp <= end);
            }

            fetchedEvents.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));

            setEvents(fetchedEvents);
            // await logEvent('analytics_dashboard_view', { user_id: currentUserId, status: 'success', filters: { eventType: filterEventType, userId: filterUserId, startDate: filterStartDate, endDate: filterEndDate } });
        } catch (err) {
            console.error("Error fetching analytics events:", err);
            setError(err.message || "Failed to fetch analytics events.");
            // await logEvent('analytics_dashboard_view', { user_id: currentUserId, status: 'failure', error: err.message });
        } finally {
            setLoading(false);
        }
    }, [db, appId, currentUserId, filterEventType, filterUserId, filterStartDate, filterEndDate]);

    useEffect(() => {
        fetchAnalyticsEvents();
    }, [fetchAnalyticsEvents]);

    const handleFilterChange = (setter) => (e) => {
        setter(e.target.value);
    };

    const handleApplyFilters = () => {
        fetchAnalyticsEvents();
    };

    const handleExport = () => {
        const headers = ["Event ID", "Timestamp", "User ID", "Event Type", "Tool Name", "Success", "Error Message", "Details (JSON)"];
        const csvRows = [headers.join(',')];

        events.forEach(event => {
            const row = [
                event.id,
                event.timestamp ? new Date(event.timestamp).toISOString() : '',
                event.user_id || 'N/A',
                event.event_type || 'N/A',
                event.details?.tool_name || 'N/A',
                event.success ? 'True' : 'False',
                event.error_message || '',
                JSON.stringify(event.details || {})
            ];
            csvRows.push(row.map(item => `"${String(item).replace(/"/g, '""')}"`).join(','));
        });

        const csvString = csvRows.join('\n');
        const blob = new Blob([csvString], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.setAttribute('download', `analytics_data_${new Date().toISOString().split('T')[0]}.csv`);
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };

    const uniqueEventTypes = [...new Set(events.map(e => e.event_type))];
    const uniqueUserIds = [...new Set(events.map(e => e.user_id))];

    return (
        <div className="analytics-dashboard-container">
            <div className="dashboard-header">
                <h2 className="dashboard-title">Analytics Dashboard</h2>
                <a href="/docs/ADMIN_GUIDE.md" target="_blank" rel="noopener noreferrer" className="admin-guide-button">
                    Admin Guide
                </a>
            </div>

            <div className="filters-grid">
                <select
                    value={filterEventType}
                    onChange={handleFilterChange(setFilterEventType)}
                    className="filter-select"
                >
                    <option value="">All Event Types</option>
                    {uniqueEventTypes.map(type => <option key={type} value={type}>{type}</option>)}
                </select>
                <select
                    value={filterUserId}
                    onChange={handleFilterChange(setFilterUserId)}
                    className="filter-select"
                >
                    <option value="">All User IDs</option>
                    {uniqueUserIds.map(id => <option key={id} value={id}>{id}</option>)}
                </select>
                <input
                    type="date"
                    value={filterStartDate}
                    onChange={handleFilterChange(setFilterStartDate)}
                    className="filter-input"
                />
                <input
                    type="date"
                    value={filterEndDate}
                    onChange={handleFilterChange(setFilterEndDate)}
                    className="filter-input"
                />
                <button
                    onClick={handleApplyFilters}
                    className="apply-filters-button"
                >
                    Apply Filters
                </button>
                 <button
                    onClick={handleExport}
                    className="export-button"
                >
                    Export CSV
                </button>
            </div>

            {loading ? (
                <div className="loading-message">Loading analytics data...</div>
            ) : error ? (
                <div className="error-message">Error: {error}</div>
            ) : events.length === 0 ? (
                <p className="no-events-message">No analytics events found for the selected filters.</p>
            ) : (
                <div className="table-container">
                    <table className="analytics-table">
                        <thead className="table-header">
                            <tr>
                                <th>Timestamp</th>
                                <th>Event Type</th>
                                <th>User ID</th>
                                <th>Details</th>
                                <th>Success</th>
                                <th>Error Message</th>
                            </tr>
                        </thead>
                        <tbody>
                            {events.map((event) => (
                                <tr key={event.id} className="table-row">
                                    <td>{new Date(event.timestamp).toLocaleString()}</td>
                                    <td>{event.event_type}</td>
                                    <td>{event.user_id}</td>
                                    <td>
                                        <pre>{JSON.stringify(event.details, null, 2)}</pre>
                                    </td>
                                    <td>
                                        {event.success === true && <span className="success-true">True</span>}
                                        {event.success === false && <span className="success-false">False</span>}
                                        {event.success === null && <span className="success-null">N/A</span>}
                                    </td>
                                    <td>
                                        <pre>{event.error_message || 'N/A'}</pre>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}
        </div>
    );
};

export default AnalyticsDashboard;

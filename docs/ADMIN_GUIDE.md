# Admin Guide: Intelli-Agent Application

## 1. Introduction

This guide is for administrators of the Intelli-Agent application. It provides detailed instructions on how to manage users, monitor application activity, and configure settings. To access the admin features, you must have an account with the "admin" role.

## 2. Accessing the Admin Dashboard

1.  Log in with your administrator account.
2.  Click on the "Admin" link in the navigation bar.
3.  This will take you to the Admin Dashboard.

## 3. Managing Users

The Admin Dashboard provides a list of all users in the system. From here, you can:

*   **View user profiles:** Click on a user to view their profile details.
*   **Change user roles:** You can assign or revoke the "admin" role for other users.
*   **Manage user status:** You can activate, disable, or suspend user accounts.


### 3.1. Managing Storage Limits

The "User Storage Management" section of the Admin Dashboard allows you to manage storage quotas for individual users.

*   **View Storage Usage:** You can see each user's current storage usage and their storage limit.
*   **Update Storage Limit:** You can enter a new storage limit (in MB) for a user and it will be saved when you click away from the input field. This allows you to override the default limit for their tier.

    *   **Assign Admin Role:** You can grant admin privileges to a user by clicking the "Make Admin" button next to their name.


*(Note: The UI for these features will need to be implemented in the Admin Dashboard.)*



## 4. Monitoring Application Analytics

The Admin Dashboard includes a powerful analytics view that allows you to monitor application activity in real-time.

### 4.1. Viewing Analytics Events

The main table displays a log of all analytics events, including:


*   User logins and registrations
*   Tool usage
*   Page views
*   Document uploads and deletions
*   Storage usage checks


*   User logins
*   User registrations
*   Tool usage
*   Page views


*   Errors

### 4.2. Filtering Analytics

You can filter the analytics data to find specific information:

*   **Filter by Event Type:** Select an event type from the dropdown to see only events of that type.
*   **Filter by User ID:** Enter a user ID to see all events for a specific user.
*   **Filter by Date Range:** Select a start and end date to see all events within that time period.

### 4.3. Exporting Data

You can export the current view of the analytics data to a CSV file by clicking the "Export CSV" button.

## 5. Configuring Administrative Users

To create a new administrator, you will need to have access to the Firestore database directly.

1.  **Create a new user:** Have the user create a new account through the application's registration page.
2.  **Find the user in Firestore:** In the Firestore console, navigate to the `users` collection and find the document corresponding to the new user.
3.  **Add the "admin" role:** In the user's document, find the `roles` array and add the string `"admin"` to it.

The user will now have administrator privileges when they next log in.

## 6. Troubleshooting

*   **User cannot access admin dashboard:** Ensure that the user's account has the "admin" role in Firestore.
*   **Analytics data not appearing:** Check the browser's developer console for any errors related to Firebase or the analytics service. Ensure that the application is properly configured to send analytics events.

For further assistance, please refer to the Owner's Guide or contact the development team.

import logging

def handle_error(error: Exception, user_message: str):
    """
    Handles an error by logging it and preparing a user-friendly message.

    :param error: The exception that was raised.
    :param user_message: The message to be displayed to the user.
    """
    logging.error(f"An error occurred: {error}", exc_info=True)
    # In a real application, you would also send a notification to the user
    # using a service like Sentry, or a custom notification system.
    print(f"User notification: {user_message}")

import unittest
import logging
from io import StringIO
from utils.error_handler import handle_error

class TestErrorHandler(unittest.TestCase):

    def test_handle_error(self):
        # Create a string buffer to capture the log output
        log_stream = StringIO()
        logging.basicConfig(stream=log_stream, level=logging.ERROR)

        # Create a string buffer to capture the print output
        print_stream = StringIO()
        import sys
        original_stdout = sys.stdout
        sys.stdout = print_stream

        # Call the function with a test error and message
        error = ValueError("Test error")
        user_message = "A test error occurred."
        handle_error(error, user_message)

        # Restore the original stdout
        sys.stdout = original_stdout

        # Assert that the error was logged
        log_output = log_stream.getvalue()
        self.assertIn("An error occurred: Test error", log_output)

        # Assert that the user message was printed
        print_output = print_stream.getvalue()
        self.assertIn("User notification: A test error occurred.", print_output)

if __name__ == '__main__':
    unittest.main()

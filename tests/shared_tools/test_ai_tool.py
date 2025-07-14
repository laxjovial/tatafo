import unittest
from unittest.mock import patch, MagicMock
from shared_tools.ai_tool import get_ai_insight

class TestAITool(unittest.TestCase):

    @patch('shared_tools.ai_tool.client.chat.completions.create')
    def test_get_ai_insight(self, mock_create):
        # Mock the OpenAI API response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "This is a test insight."
        mock_create.return_value = mock_response

        # Call the function with test data
        data = {"test_key": "test_value"}
        prompt = "Test prompt"
        result = get_ai_insight(data, prompt)

        # Assert that the function returns the expected result
        self.assertEqual(result, "This is a test insight.")

if __name__ == '__main__':
    unittest.main()

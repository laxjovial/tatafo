import unittest
from unittest.mock import patch, MagicMock
from shared_tools.llm_pipeline import execute_pipeline

class TestLlmPipeline(unittest.TestCase):

    @patch('shared_tools.llm_pipeline.get_ai_insight')
    @patch('shared_tools.llm_pipeline.finance_get_historical_stock_prices')
    def test_execute_pipeline_finance(self, mock_finance_tool, mock_get_ai_insight):
        # Mock the AI's response to identify the tool
        mock_get_ai_insight.side_effect = [
            '{"tool": "finance_get_historical_stock_prices", "params": {"ticker": "AAPL"}}',
            "The stock price for AAPL is $150."
        ]
        # Mock the finance tool's response
        mock_finance_tool.return_value = {"ticker": "AAPL", "price": 150}

        # Call the pipeline with a test query
        query = "What is the stock price of Apple?"
        result = execute_pipeline(query)

        # Assert that the correct tool was called and the final response is correct
        mock_finance_tool.assert_called_once_with(ticker="AAPL")
        self.assertEqual(result, "The stock price for AAPL is $150.")

if __name__ == '__main__':
    unittest.main()

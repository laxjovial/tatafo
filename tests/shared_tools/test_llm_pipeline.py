import unittest
from unittest.mock import patch, MagicMock, AsyncMock
# Import UserProfile for mocking the user_context
from backend.models.user_models import UserProfile
from shared_tools.llm_pipeline import execute_pipeline # Assuming this is the main function to test

class TestLlmPipeline(unittest.TestCase):

    # Patch the get_ai_insight function
    @patch('shared_tools.llm_pipeline.get_ai_insight')
    # Patch the specific method on the FinanceTools class that llm_pipeline would call
    @patch('shared_tools.llm_pipeline.FinanceTools.finance_get_historical_stock_prices', new_callable=AsyncMock)
    def test_execute_pipeline_finance(self, mock_finance_tool_method, mock_get_ai_insight):
        # Create a mock UserProfile as it's often a required argument for tool calls
        mock_user_profile = UserProfile(user_id="test_user", tier="free", roles=["user"])

        # Mock the AI's response to identify the tool and its parameters
        # The AI's response should match what execute_pipeline expects for tool calling
        mock_get_ai_insight.side_effect = [
            # First call: AI identifies the tool and its parameters
            '{"tool": "finance_get_historical_stock_prices", "params": {"symbol": "AAPL", "start_date": "2023-01-01", "end_date": "2023-01-31"}}',
            # Second call: AI generates the final response after tool execution
            "The historical stock price for AAPL from 2023-01-01 to 2023-01-31 is $150."
        ]
        
        # Mock the finance tool's response when it's called by the pipeline
        # This is what mock_finance_tool_method will return
        mock_finance_tool_method.return_value = "Historical prices for AAPL:\n  Date: 2023-01-01\n    Close: 150.0"

        # Call the pipeline with a test query
        query = "What was the stock price of Apple in January 2023?"
        result = execute_pipeline(query, user_context=mock_user_profile) # Pass mock_user_profile

        # Assert that the correct tool method was called with the expected arguments
        # The arguments should match the signature of finance_get_historical_stock_prices
        mock_finance_tool_method.assert_called_once_with(
            symbol="AAPL",
            start_date="2023-01-01",
            end_date="2023-01-31",
            user_context=mock_user_profile, # Ensure user_context is passed
            provider="alphavantage", # Default value, ensure it's included if not mocked away
            user_api_keys=[] # Default value, ensure it's included if not mocked away
        )
        
        # Assert that the final response from the pipeline is correct
        self.assertEqual(result, "The historical stock price for AAPL from 2023-01-01 to 2023-01-31 is $150.")

if __name__ == '__main__':
    unittest.main()

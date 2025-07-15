import pytest
from unittest.mock import patch, AsyncMock
from backend.models.user_models import UserProfile
from domain_tools.finance_tools.finance_tool import FinanceTools
from domain_tools.crypto_tools.crypto_tool import CryptoTools
from domain_tools.entertainment_tools.entertainment_tool import EntertainmentTools

@pytest.fixture
def user_profile():
    return UserProfile(user_id="test_user", tier="premium", roles=["user"])

@pytest.mark.asyncio
@patch('shared_tools.historical_data_tool.make_api_request', new_callable=AsyncMock)
async def test_finance_get_stock_price(mock_make_api_request, user_profile):
    mock_make_api_request.return_value = {"price": 150.0, "currency": "USD", "timestamp": "2023-10-27"}
    finance_tools = FinanceTools(config_manager=None, firestore_manager=None, log_event=None, document_tools=None)
    result = await finance_tools.finance_get_stock_price("AAPL", user_context=user_profile)
    assert "The current price of AAPL is 150.0 USD" in result

@pytest.mark.asyncio
@patch('shared_tools.historical_data_tool.make_api_request', new_callable=AsyncMock)
async def test_crypto_get_crypto_price(mock_make_api_request, user_profile):
    mock_make_api_request.return_value = {"bitcoin": {"usd": 40000.0}}
    crypto_tools = CryptoTools(config_manager=None, firestore_manager=None, log_event=None, document_tools=None)
    result = await crypto_tools.crypto_get_crypto_price("bitcoin", user_context=user_profile)
    assert "'bitcoin': {'usd': 40000.0}" in result

@pytest.mark.asyncio
@patch('domain_tools.entertainment_tools.entertainment_tool.EntertainmentTools._make_dynamic_api_request', new_callable=AsyncMock)
async def test_entertainment_search_movies(mock_make_dynamic_api_request, user_profile):
    mock_make_dynamic_api_request.return_value = {
        "title": "Inception",
        "year": "2010",
        "genre": "Sci-Fi",
        "director": "Christopher Nolan",
        "plot": "A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O.",
        "imdb_rating": "8.8",
        "poster": "https://example.com/poster.jpg"
    }
    entertainment_tools = EntertainmentTools(config_manager=None, log_event=None, document_tools=None)
    result = await entertainment_tools.entertainment_search_movies("Inception", user_context=user_profile)
    assert "Movie: Inception (2010)" in result

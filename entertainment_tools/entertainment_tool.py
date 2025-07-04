# domain_tools/entertainment_tools/entertainment_tool.py

import logging
from typing import Optional, Dict, Any, List
from langchain_core.tools import tool

# Import config_manager for API keys
from config.config_manager import config_manager
# Import user_manager for RBAC checks
from utils.user_manager import get_user_tier_capability

logger = logging.getLogger(__name__)

# --- Helper Function to get API Keys for Entertainment APIs ---
def _get_entertainment_api_key(api_name: str) -> Optional[str]:
    """
    Retrieves the API key for a given entertainment API from secrets.
    """
    if api_name == "themoviedb": # Example placeholder for a real API
        return config_manager.get_secret("themoviedb_api_key")
    if api_name == "spotify": # Example placeholder for a real API
        return config_manager.get_secret("spotify_api_key")
    # Add other entertainment API key retrieval logic here if needed
    return None

@tool
def get_movie_details(movie_title: str, user_token: str = "default") -> str:
    """
    Retrieves detailed information about a movie.
    Uses a mock entertainment API for demonstration.

    Args:
        movie_title (str): The title of the movie (e.g., "Inception", "Dune", "The Matrix").
        user_token (str, optional): The unique identifier for the user. Defaults to "default".
                                    Used for RBAC capability checks.

    Returns:
        str: A string containing the movie details, or an error message.
    """
    logger.info(f"Tool: get_movie_details called for movie: {movie_title} by user: {user_token}")

    if not get_user_tier_capability(user_token, 'entertainment_tool_access', False):
        return "Error: Access to entertainment tools is not enabled for your current tier."

    # In a real application, you would make an API call here (e.g., TMDb API, OMDb API).
    # For demonstration, we'll use mock data.
    mock_movie_data = {
        "inception": {
            "title": "Inception",
            "director": "Christopher Nolan",
            "year": 2010,
            "genre": "Science Fiction, Action, Thriller",
            "plot": "A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O.",
            "rating": 8.8
        },
        "dune": {
            "title": "Dune",
            "director": "Denis Villeneuve",
            "year": 2021,
            "genre": "Science Fiction, Adventure, Drama",
            "plot": "A noble young man is thrust into a galactic war over the most important resource in the universe.",
            "rating": 8.0
        },
        "the matrix": {
            "title": "The Matrix",
            "director": "The Wachowskis",
            "year": 1999,
            "genre": "Science Fiction, Action",
            "plot": "A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers.",
            "rating": 8.7
        }
    }

    movie_info = mock_movie_data.get(movie_title.lower())

    if movie_info:
        formatted_details = (
            f"Movie Details for '{movie_info['title']}':\n"
            f"Director: {movie_info['director']}\n"
            f"Year: {movie_info['year']}\n"
            f"Genre: {movie_info['genre']}\n"
            f"Plot: {movie_info['plot']}\n"
            f"Rating: {movie_info['rating']}/10"
        )
        return formatted_details
    else:
        return f"Movie details not found for '{movie_title}'. Please check the spelling or try a different movie."

@tool
def get_music_artist_info(artist_name: str, user_token: str = "default") -> str:
    """
    Retrieves information about a music artist.
    Uses a mock entertainment API for demonstration.

    Args:
        artist_name (str): The name of the music artist (e.g., "Taylor Swift", "Queen", "The Beatles").
        user_token (str, optional): The unique identifier for the user. Defaults to "default".
                                    Used for RBAC capability checks.

    Returns:
        str: A string containing the artist information, or an error message.
    """
    logger.info(f"Tool: get_music_artist_info called for artist: {artist_name} by user: {user_token}")

    if not get_user_tier_capability(user_token, 'entertainment_tool_access', False):
        return "Error: Access to entertainment tools is not enabled for your current tier."

    # In a real application, you would make an API call here (e.g., Spotify API, MusicBrainz API).
    # For demonstration, we'll use mock data.
    mock_artist_data = {
        "taylor swift": {
            "name": "Taylor Swift",
            "genre": "Pop, Country",
            "notable_albums": ["Fearless", "1989", "Folklore", "Midnights"],
            "bio_snippet": "An American singer-songwriter. Her narrative songwriting, often inspired by her personal life, has received widespread critical praise and media coverage."
        },
        "queen": {
            "name": "Queen",
            "genre": "Rock, Glam Rock, Hard Rock",
            "notable_albums": ["A Night at the Opera", "News of the World", "The Game"],
            "bio_snippet": "A British rock band formed in London in 1970. Their classic line-up was Freddie Mercury (lead vocals, piano), Brian May (guitar, vocals), John Deacon (bass guitar), and Roger Taylor (drums, vocals)."
        },
        "the beatles": {
            "name": "The Beatles",
            "genre": "Rock, Pop, Beat",
            "notable_albums": ["Sgt. Pepper's Lonely Hearts Club Band", "Abbey Road", "Revolver"],
            "bio_snippet": "An English rock band, formed in Liverpool in 1960, who became the most commercially successful and critically acclaimed act in the history of music."
        }
    }

    artist_info = mock_artist_data.get(artist_name.lower())

    if artist_info:
        formatted_details = (
            f"Music Artist Information for '{artist_info['name']}':\n"
            f"Genre: {artist_info['genre']}\n"
            f"Notable Albums: {', '.join(artist_info['notable_albums'])}\n"
            f"Bio: {artist_info['bio_snippet']}"
        )
        return formatted_details
    else:
        return f"Music artist information not found for '{artist_name}'. Please check the spelling or try a different artist."

# CLI Test (optional)
if __name__ == "__main__":
    import sys
    from unittest.mock import MagicMock, patch

    logging.basicConfig(level=logging.INFO)

    # Mock Streamlit secrets and config_manager for local testing
    class MockSecrets:
        def __init__(self):
            self.themoviedb_api_key = "MOCK_TMDB_KEY"
            self.spotify_api_key = "MOCK_SPOTIFY_KEY"
            self.openai = {"api_key": "sk-mock-openai-key-12345"}
            self.google = {"api_key": "AIzaSy-mock-google-key"}
            self.firebase_config = "{}"

        def get(self, key, default=None):
            parts = key.split('.')
            val = self
            for part in parts:
                if hasattr(val, part):
                    val = getattr(val, part)
                elif isinstance(val, dict) and part in val:
                    val = val[part]
                else:
                    return default
            return val
    
    class MockConfigManager:
        _instance = None
        _is_loaded = False
        def __init__(self):
            if MockConfigManager._instance is not None:
                raise Exception("ConfigManager is a singleton. Use get_instance().")
            MockConfigManager._instance = self
            self._config_data = {
                'llm': {'max_summary_input_chars': 10000},
                'rag': {'chunk_size': 500, 'chunk_overlap': 50, 'max_query_results_k': 10},
                'web_scraping': {
                    'user_agent': 'Mozilla/5.0 (Test; Python)',
                    'timeout_seconds': 1 # Short timeout for mocks
                },
                'tiers': {},
                'default_user_tier': 'free',
                'default_user_roles': ['user'],
                'api_configs': []
            }
            self._is_loaded = True
        
        def get(self, key, default=None):
            parts = key.split('.')
            val = self._config_data
            for part in parts:
                if isinstance(val, dict) and part in val:
                    val = val[part]
                else:
                    return default
            return val
        
        def get_secret(self, key, default=None):
            if key == "themoviedb_api_key": return st.secrets.themoviedb_api_key
            if key == "spotify_api_key": return st.secrets.spotify_api_key
            return st.secrets.get(key, default)

        def set_secret(self, key, value):
            setattr(st.secrets, key, value)


    # Mock user_manager.get_current_user and get_user_tier_capability for testing RBAC
    class MockUserManager:
        _mock_users = {
            "mock_free_token": {"user_id": "mock_free_token", "username": "FreeUser", "email": "free@example.com", "tier": "free", "roles": ["user"]},
            "mock_pro_token": {"user_id": "mock_pro_token", "username": "ProUser", "email": "pro@example.com", "tier": "pro", "roles": ["user"]},
            "mock_premium_token": {"user_id": "mock_premium_token", "username": "PremiumUser", "email": "premium@example.com", "tier": "premium", "roles": ["user"]},
            "mock_admin_token": {"user_id": "mock_admin_token", "username": "AdminUser", "email": "admin@example.com", "tier": "admin", "roles": ["user", "admin"]},
        }
        _rbac_capabilities = {
            'capabilities': {
                'entertainment_tool_access': {
                    'default': False,
                    'roles': {'pro': True, 'premium': True, 'admin': True} # Often entertainment features are not free
                }
            }
        }
        _tier_hierarchy = {
            "free": 0, "user": 1, "basic": 2, "pro": 3, "premium": 4, "admin": 99
        }

        def get_current_user(self) -> Dict[str, Any]:
            return getattr(self, '_current_mock_user', {})

        def get_user_tier_capability(self, user_token: Optional[str], capability_key: str, default_value: Any = None) -> Any:
            user_info = self._mock_users.get(user_token, {})
            user_id = user_info.get('user_id')
            user_tier = user_info.get('tier', 'free')
            user_roles = user_info.get('roles', [])

            if "admin" in user_roles:
                if isinstance(default_value, bool): return True
                if isinstance(default_value, (int, float)): return float('inf')
                return default_value
            
            capability_config = self._rbac_capabilities.get('capabilities', {}).get(capability_key)
            if not capability_config:
                return default_value

            for role in user_roles:
                if role in capability_config.get('roles', {}):
                    return capability_config['roles'][role]
            
            return capability_config.get('default', default_value)

    # Patch the actual imports for testing
    import streamlit as st_mock
    if not hasattr(st_mock, 'secrets'):
        st_mock.secrets = MockSecrets()
    
    sys.modules['config.config_manager'].config_manager = MockConfigManager()
    sys.modules['config.config_manager'].ConfigManager = MockConfigManager
    sys.modules['utils.user_manager'] = MockUserManager()
    sys.modules['utils.user_manager']._RBAC_CAPABILITIES = MockUserManager()._rbac_capabilities
    sys.modules['utils.user_manager']._TIER_HIERARCHY = MockUserManager()._tier_hierarchy

    # Mock requests.get for external API calls (not strictly needed for this mock, but good practice)
    original_requests_get = requests.get
    requests.get = MagicMock() # Mock all requests.get calls

    test_user_free = sys.modules['utils.user_manager']._mock_users["mock_free_token"]['user_id']
    test_user_pro = sys.modules['utils.user_manager']._mock_users["mock_pro_token"]['user_id']
    test_user_premium = sys.modules['utils.user_manager']._mock_users["mock_premium_token"]['user_id']
    test_user_admin = sys.modules['utils.user_manager']._mock_users["mock_admin_token"]['user_id']

    print("\n--- Testing get_movie_details function ---")

    # Test 1: Pro user with access, valid movie (Inception)
    print("\n--- Test 1: Pro user with access, valid movie (Inception) ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_pro
    result1 = get_movie_details("Inception", user_token=test_user_pro)
    print(f"Result for Inception (Pro User):\n{result1[:100]}...")
    assert "Movie Details for 'Inception':" in result1
    assert "Director: Christopher Nolan" in result1
    print("Test 1 Passed.")

    # Test 2: Free user, access denied
    print("\n--- Test 2: Free user, access denied ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_free
    result2 = get_movie_details("Dune", user_token=test_user_free)
    print(f"Result for Dune (Free user): {result2}")
    assert "Error: Access to entertainment tools is not enabled for your current tier." in result2
    print("Test 2 Passed.")

    # Test 3: Admin user, movie not found
    print("\n--- Test 3: Admin user, movie not found ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_admin
    result3 = get_movie_details("Avatar 2", user_token=test_user_admin)
    print(f"Result for Avatar 2 (Admin user): {result3}")
    assert "Movie details not found for 'Avatar 2'." in result3
    print("Test 3 Passed.")

    print("\n--- Testing get_music_artist_info function ---")

    # Test 4: Premium user with access, valid artist (Taylor Swift)
    print("\n--- Test 4: Premium user with access, valid artist (Taylor Swift) ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_premium
    result4 = get_music_artist_info("Taylor Swift", user_token=test_user_premium)
    print(f"Result for Taylor Swift (Premium user):\n{result4[:100]}...")
    assert "Music Artist Information for 'Taylor Swift':" in result4
    assert "Genre: Pop, Country" in result4
    print("Test 4 Passed.")

    # Test 5: Free user, access denied
    print("\n--- Test 5: Free user, access denied ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_free
    result5 = get_music_artist_info("Queen", user_token=test_user_free)
    print(f"Result for Queen (Free user): {result5}")
    assert "Error: Access to entertainment tools is not enabled for your current tier." in result5
    print("Test 5 Passed.")

    # Test 6: Pro user, artist not found
    print("\n--- Test 6: Pro user, artist not found ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_pro
    result6 = get_music_artist_info("Led Zeppelin", user_token=test_user_pro)
    print(f"Result for Led Zeppelin (Pro user): {result6}")
    assert "Music artist information not found for 'Led Zeppelin'." in result6
    print("Test 6 Passed.")

    print("\nAll entertainment_tool tests passed (mocked data and RBAC).")

    # Restore original requests.get
    requests.get = original_requests_get

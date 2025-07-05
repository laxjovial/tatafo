# domain_tools/sports_tools/sports_tool.py

import logging
from typing import Optional, Dict, Any, List
from langchain_core.tools import tool
from datetime import datetime, timedelta

# Import config_manager for API keys
from config.config_manager import config_manager
# Import user_manager for RBAC checks
from utils.user_manager import get_user_tier_capability

logger = logging.getLogger(__name__)

# --- Helper Function to get API Keys for Sports APIs ---
def _get_sports_api_key(api_name: str) -> Optional[str]:
    """
    Retrieves the API key for a given sports API from secrets.
    """
    if api_name == "sportradar": # Example placeholder for a real API
        return config_manager.get_secret("sportradar_api_key")
    if api_name == "thesportsdb": # Example placeholder for a real API
        return config_manager.get_secret("thesportsdb_api_key")
    # Add other sports API key retrieval logic here if needed
    return None

# --- Mock Data for Demonstration (Comprehensive) ---
MOCK_SPORTS_DATA = {
    "players": {
        "lewis hamilton": {
            "name": "Lewis Hamilton",
            "sport": "Formula 1",
            "team": "Mercedes-AMG Petronas F1 Team",
            "nationality": "British",
            "stats": {"championships_won": 7, "race_wins": 103, "pole_positions": 104},
            "trophies": ["7x Formula 1 World Champion"],
            "position": "Driver",
            "rings_won": "N/A", # Not applicable for F1
            "titles_won": ["Sir", "MBE"],
            "championship_stats": {"2020": {"points": 347, "wins": 11}}
        },
        "lionel messi": {
            "name": "Lionel Messi",
            "sport": "Football (Soccer)",
            "club": "Inter Miami CF",
            "nationality": "Argentinian",
            "stats": {"goals": 838, "assists": 372, "ballon_dor": 8},
            "trophies": ["1x FIFA World Cup", "4x UEFA Champions League", "10x La Liga", "8x Ballon d'Or"],
            "position": "Forward",
            "rings_won": "N/A", # Not applicable for Football
            "titles_won": ["World Cup Winner", "Copa América Winner"],
            "championship_stats": {"2022_world_cup": {"goals": 7, "assists": 3, "golden_ball": True}}
        },
        "lebron james": {
            "name": "LeBron James",
            "sport": "Basketball (NBA)",
            "team": "Los Angeles Lakers",
            "nationality": "American",
            "stats": {"points": 40474, "assists": 11196, "rebounds": 11210}, # As of early 2024
            "trophies": ["4x NBA Champion", "4x NBA Finals MVP", "4x NBA MVP"],
            "position": "Small Forward",
            "rings_won": "4",
            "titles_won": ["King James", "All-Star MVP"],
            "championship_stats": {"2020_nba_finals": {"ppg": 29.8, "apg": 8.5, "rpg": 11.8}}
        },
        "conor mcgregor": {
            "name": "Conor McGregor",
            "sport": "MMA (UFC)",
            "team": "SBG Ireland",
            "nationality": "Irish",
            "stats": {"wins": 22, "losses": 6, "knockouts": 19},
            "trophies": ["UFC Featherweight Champion", "UFC Lightweight Champion"],
            "position": "Lightweight, Featherweight",
            "rings_won": "N/A", # Not applicable for MMA
            "titles_won": ["The Notorious"],
            "championship_stats": {"ufc_205": {"round": 2, "method": "KO"}}
        }
    },
    "teams": {
        "real madrid": {
            "name": "Real Madrid CF",
            "sport": "Football (Soccer)",
            "league": "La Liga",
            "nationality": "Spanish",
            "stats": {"wins": 24, "draws": 8, "losses": 4, "goals_for": 87, "goals_against": 26}, # Example season stats
            "trophies": ["15x UEFA Champions League", "36x La Liga", "20x Copa del Rey"],
            "current_standing": "1st in La Liga (2023-2024 season)"
        },
        "golden state warriors": {
            "name": "Golden State Warriors",
            "sport": "Basketball (NBA)",
            "league": "NBA",
            "nationality": "American",
            "stats": {"wins": 46, "losses": 36, "win_percentage": 0.561}, # Example season stats
            "trophies": ["7x NBA Championship"],
            "current_standing": "6th in Western Conference (2023-2024 season)"
        }
    },
    "leagues": {
        "premier league": {
            "name": "Premier League",
            "sport": "Football (Soccer)",
            "country": "England",
            "current_champion": "Manchester City (2023-2024)",
            "most_titles_club": "Manchester United (20)",
            "top_scorers_2023-2024": ["Erling Haaland (27)", "Cole Palmer (22)"]
        },
        "nba": {
            "name": "National Basketball Association (NBA)",
            "sport": "Basketball",
            "country": "USA/Canada",
            "current_champion": "Denver Nuggets (2023)",
            "most_titles_team": "Boston Celtics (17), Los Angeles Lakers (17)",
            "mvp_2023": "Nikola Jokic"
        }
    }
}

@tool
def get_player_stats(player_name: str, sport: Optional[str] = None, user_token: str = "default") -> str:
    """
    Retrieves detailed statistics and achievements for a specific player across various sports.
    This includes career stats, trophies, position, rings won (for basketball), and specific championship stats.

    Args:
        player_name (str): The full name of the player (e.g., "Lionel Messi", "LeBron James").
        sport (str, optional): The specific sport if known (e.g., "Football", "Basketball", "Formula 1", "MMA").
                               Helps narrow down search for common names.
        user_token (str, optional): The unique identifier for the user. Defaults to "default".

    Returns:
        str: A formatted string with the player's information, or an error message.
    """
    logger.info(f"Tool: get_player_stats called for player: {player_name}, sport: {sport} by user: {user_token}")

    if not get_user_tier_capability(user_token, 'sports_tool_access', False):
        return "Error: Access to sports information tools is not enabled for your current tier."

    norm_player_name = player_name.lower()
    norm_sport = sport.lower() if sport else None

    found_player = None
    for p_key, p_data in MOCK_SPORTS_DATA["players"].items():
        if norm_player_name in p_key or p_key in norm_player_name:
            if norm_sport and norm_sport not in p_data["sport"].lower():
                continue # Skip if sport doesn't match
            found_player = p_data
            break

    if found_player:
        stats_str = ", ".join([f"{k.replace('_', ' ').title()}: {v}" for k, v in found_player["stats"].items()])
        trophies_str = ", ".join(found_player["trophies"]) if found_player["trophies"] else "None listed."
        championship_stats_str = ""
        if found_player["championship_stats"]:
            championship_stats_str = "\nChampionship Stats (Key Events):\n"
            for year, c_stats in found_player["championship_stats"].items():
                c_stats_formatted = ", ".join([f"{k.replace('_', ' ').title()}: {v}" for k, v in c_stats.items()])
                championship_stats_str += f"  {year}: {c_stats_formatted}\n"

        return (
            f"Player Stats for {found_player['name']} ({found_player['sport']}):\n"
            f"  Team/Club: {found_player.get('team') or found_player.get('club')}\n"
            f"  Nationality: {found_player['nationality']}\n"
            f"  Position: {found_player['position']}\n"
            f"  Career Stats: {stats_str}\n"
            f"  Trophies Won: {trophies_str}\n"
            f"  Rings Won (Basketball): {found_player['rings_won']}\n"
            f"  Titles Won: {', '.join(found_player['titles_won']) if found_player['titles_won'] else 'None listed.'}"
            f"{championship_stats_str}"
        )
    else:
        return f"Player '{player_name}' not found or no data available. Please check spelling or specify sport."

@tool
def get_team_stats(team_name: str, sport: Optional[str] = None, user_token: str = "default") -> str:
    """
    Retrieves detailed statistics and achievements for a specific sports team/club.
    This includes recent season stats, major trophies, and current league standing.

    Args:
        team_name (str): The full name of the team/club (e.g., "Real Madrid", "Golden State Warriors").
        sport (str, optional): The specific sport if known (e.g., "Football", "Basketball").
                               Helps narrow down search.
        user_token (str, optional): The unique identifier for the user. Defaults to "default".

    Returns:
        str: A formatted string with the team's information, or an error message.
    """
    logger.info(f"Tool: get_team_stats called for team: {team_name}, sport: {sport} by user: {user_token}")

    if not get_user_tier_capability(user_token, 'sports_tool_access', False):
        return "Error: Access to sports information tools is not enabled for your current tier."

    norm_team_name = team_name.lower()
    norm_sport = sport.lower() if sport else None

    found_team = None
    for t_key, t_data in MOCK_SPORTS_DATA["teams"].items():
        if norm_team_name in t_key or t_key in norm_team_name:
            if norm_sport and norm_sport not in t_data["sport"].lower():
                continue # Skip if sport doesn't match
            found_team = t_data
            break

    if found_team:
        stats_str = ", ".join([f"{k.replace('_', ' ').title()}: {v}" for k, v in found_team["stats"].items()])
        trophies_str = ", ".join(found_team["trophies"]) if found_team["trophies"] else "None listed."
        
        return (
            f"Team Stats for {found_team['name']} ({found_team['sport']}):\n"
            f"  League: {found_team['league']}\n"
            f"  Nationality: {found_team['nationality']}\n"
            f"  Recent Season Stats: {stats_str}\n"
            f"  Major Trophies: {trophies_str}\n"
            f"  Current Standing: {found_team['current_standing']}"
        )
    else:
        return f"Team '{team_name}' not found or no data available. Please check spelling or specify sport."

@tool
def get_league_info(league_name: str, sport: Optional[str] = None, user_token: str = "default") -> str:
    """
    Retrieves information about a specific sports league.
    This includes current champion, most successful clubs/teams, and top performers.

    Args:
        league_name (str): The name of the league (e.g., "Premier League", "NBA").
        sport (str, optional): The specific sport if known (e.g., "Football", "Basketball").
                               Helps narrow down search.
        user_token (str, optional): The unique identifier for the user. Defaults to "default".

    Returns:
        str: A formatted string with the league's information, or an error message.
    """
    logger.info(f"Tool: get_league_info called for league: {league_name}, sport: {sport} by user: {user_token}")

    if not get_user_tier_capability(user_token, 'sports_tool_access', False):
        return "Error: Access to sports information tools is not enabled for your current tier."

    norm_league_name = league_name.lower()
    norm_sport = sport.lower() if sport else None

    found_league = None
    for l_key, l_data in MOCK_SPORTS_DATA["leagues"].items():
        if norm_league_name in l_key or l_key in norm_league_name:
            if norm_sport and norm_sport not in l_data["sport"].lower():
                continue # Skip if sport doesn't match
            found_league = l_data
            break

    if found_league:
        top_scorers_str = ", ".join(found_league.get("top_scorers_2023-2024", []))
        
        return (
            f"League Information for {found_league['name']} ({found_league['sport']}):\n"
            f"  Country: {found_league['country']}\n"
            f"  Current Champion: {found_league['current_champion']}\n"
            f"  Most Titles (Team): {found_league['most_titles_team']}\n"
            f"  Top Scorers (2023-2024): {top_scorers_str if top_scorers_str else 'N/A'}\n"
            f"  MVP (2023): {found_league.get('mvp_2023', 'N/A')}"
        )
    else:
        return f"League '{league_name}' not found or no data available. Please check spelling or specify sport."

# CLI Test (optional)
if __name__ == "__main__":
    import sys
    from unittest.mock import MagicMock, patch

    logging.basicConfig(level=logging.INFO)

    # Mock Streamlit secrets and config_manager for local testing
    class MockSecrets:
        def __init__(self):
            self.sportradar_api_key = "MOCK_SPORTRADAR_KEY"
            self.thesportsdb_api_key = "MOCK_THESPORTSDB_KEY"
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
            if key == "sportradar_api_key": return st.secrets.sportradar_api_key
            if key == "thesportsdb_api_key": return st.secrets.thesportsdb_api_key
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
                'sports_tool_access': {
                    'default': False,
                    'roles': {'pro': True, 'premium': True, 'admin': True}
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

    print("\n--- Testing get_player_stats function ---")

    # Test 1: Pro user with access, valid player (Lionel Messi)
    print("\n--- Test 1: Pro user with access, valid player (Lionel Messi) ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_pro
    result1 = get_player_stats("Lionel Messi", sport="Football", user_token=test_user_pro)
    print(f"Result for Lionel Messi (Pro User):\n{result1[:100]}...")
    assert "Player Stats for Lionel Messi (Football):" in result1
    assert "Goals: 838" in result1
    print("Test 1 Passed.")

    # Test 2: Free user, access denied
    print("\n--- Test 2: Free user, access denied ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_free
    result2 = get_player_stats("LeBron James", user_token=test_user_free)
    print(f"Result for LeBron James (Free user): {result2}")
    assert "Error: Access to sports information tools is not enabled for your current tier." in result2
    print("Test 2 Passed.")

    # Test 3: Admin user, player not found
    print("\n--- Test 3: Admin user, player not found ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_admin
    result3 = get_player_stats("Roger Federer", user_token=test_user_admin)
    print(f"Result for Roger Federer (Admin user): {result3}")
    assert "Player 'Roger Federer' not found or no data available." in result3
    print("Test 3 Passed.")

    print("\n--- Testing get_team_stats function ---")

    # Test 4: Premium user with access, valid team (Real Madrid)
    print("\n--- Test 4: Premium user with access, valid team (Real Madrid) ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_premium
    result4 = get_team_stats("Real Madrid", sport="Football", user_token=test_user_premium)
    print(f"Result for Real Madrid (Premium user):\n{result4[:100]}...")
    assert "Team Stats for Real Madrid CF (Football):" in result4
    assert "15x UEFA Champions League" in result4
    print("Test 4 Passed.")

    # Test 5: Free user, access denied
    print("\n--- Test 5: Free user, access denied ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_free
    result5 = get_team_stats("Golden State Warriors", user_token=test_user_free)
    print(f"Result for Golden State Warriors (Free user): {result5}")
    assert "Error: Access to sports information tools is not enabled for your current tier." in result5
    print("Test 5 Passed.")

    # Test 6: Pro user, team not found
    print("\n--- Test 6: Pro user, team not found ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_pro
    result6 = get_team_stats("Chicago Bulls", user_token=test_user_pro)
    print(f"Result for Chicago Bulls (Pro user): {result6}")
    assert "Team 'Chicago Bulls' not found or no data available." in result6
    print("Test 6 Passed.")

    print("\n--- Testing get_league_info function ---")

    # Test 7: Admin user with access, valid league (Premier League)
    print("\n--- Test 7: Admin user with access, valid league (Premier League) ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_admin
    result7 = get_league_info("Premier League", user_token=test_user_admin)
    print(f"Result for Premier League (Admin user):\n{result7[:100]}...")
    assert "League Information for Premier League (Football):" in result7
    assert "Current Champion: Manchester City" in result7
    print("Test 7 Passed.")

    # Test 8: Free user, access denied
    print("\n--- Test 8: Free user, access denied ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_free
    result8 = get_league_info("NBA", user_token=test_user_free)
    print(f"Result for NBA (Free user): {result8}")
    assert "Error: Access to sports information tools is not enabled for your current tier." in result8
    print("Test 8 Passed.")

    # Test 9: Premium user, league not found
    print("\n--- Test 9: Premium user, league not found ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_premium
    result9 = get_league_info("NFL", user_token=test_user_premium)
    print(f"Result for NFL (Premium user): {result9}")
    assert "League 'NFL' not found or no data available." in result9
    print("Test 9 Passed.")

    print("\nAll sports_tool tests passed (mocked data and RBAC).")

    # Restore original requests.get
    requests.get = original_requests_get

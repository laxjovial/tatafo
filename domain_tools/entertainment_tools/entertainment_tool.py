# domain_tools/entertainment_tools/entertainment_tool.py

import logging
import requests
import json
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime, timedelta, timezone # Import timezone for consistent datetime objects

# Import generic tools
from langchain_core.tools import tool
from shared_tools.scrapper_tool import scrape_web # This tool is a standalone function

# Import config_manager to access API configurations and secrets
from config.config_manager import config_manager
# Import user_manager for RBAC checks
from utils.user_manager import get_user_tier_capability
# Import date_parser for date format flexibility
from utils.date_parser import parse_date_to_yyyymmdd

# Import analytics_tracker (for logging failures within _make_dynamic_api_request)
from utils import analytics_tracker

# Import UserProfile for type hinting
from backend.models.user_models import UserProfile

logger = logging.getLogger(__name__)

# --- Generic API Request Helper (copied for standalone tool file, ideally in shared utils) ---
# This helper is designed to work with the structure defined in api_providers.yml

def _get_nested_value(data: Dict[str, Any], path: List[str]):
    """Helper to get a value from a nested dictionary using a list of keys."""
    current = data
    for key in path:
        if isinstance(current, dict) and key in current:
            current = current[key]
        elif isinstance(current, list) and isinstance(key, str) and key.isdigit(): # Handle list indices
            try:
                current = current[int(key)]
            except (IndexError, ValueError):
                return None
        else:
            return None
    return current

class EntertainmentTools:
    """
    A collection of tools for comprehensive entertainment-related operations,
    including searching for movies, TV shows, music, anime, and podcasts.
    It integrates with external APIs and provides fallback mechanisms.
    """
    def __init__(self, config_manager, log_event, document_tools):
        self.config_manager = config_manager
        self.log_event = log_event # For direct logging if needed, but _make_dynamic_api_request handles tool usage
        self.document_tools = document_tools # For entertainment_query_uploaded_docs and entertainment_summarize_document_by_path

    async def _make_dynamic_api_request(
        self,
        domain: str,
        function_name: str,
        params: Dict[str, Any],
        user_context: UserProfile # Changed from user_token to user_context
    ) -> Optional[Dict[str, Any]]:
        """
        Makes an API request to the dynamically configured provider for a given domain and function.
        Handles API key retrieval, request construction, and basic error handling.
        Returns parsed JSON data or None on failure (triggering generic fallback message).
        Logs tool usage analytics for *failures* via analytics_tracker.
        Success logging is handled by LLMService's wrapped_tool_executor.
        """
        user_id = user_context.user_id # Get user_id from UserProfile

        # Get the default active API provider for the domain from data/config.yml
        active_provider_name = self.config_manager.get(f"api_defaults.{domain}")
        if not active_provider_name:
            logger.error(f"No default API provider configured for domain '{domain}'.")
            await analytics_tracker.log_tool_usage(
                tool_name=f"{domain}_{function_name}",
                tool_params=params,
                user_id=user_id,
                success=False,
                error_message=f"No default API provider configured for domain '{domain}'."
            )
            return None

        # Get the full configuration for the active provider from api_providers.yml
        provider_config = self.config_manager.get_api_provider_config(domain, active_provider_name)
        if not provider_config:
            logger.error(f"Configuration for API provider '{active_provider_name}' in domain '{domain}' not found in api_providers.yml.")
            await analytics_tracker.log_tool_usage(
                tool_name=f"{domain}_{function_name}",
                tool_params=params,
                user_id=user_id,
                success=False,
                error_message=f"API provider config '{active_provider_name}' not found for domain '{domain}'."
            )
            return None

        base_url = provider_config.get("base_url")
        api_key_name = provider_config.get("api_key_name")
        api_key = self.config_manager.get_secret(api_key_name) if api_key_name else None

        headers = {} # No special headers by default for most entertainment APIs

        if not base_url:
            logger.error(f"Base URL not configured for API provider '{active_provider_name}' in domain '{domain}'.")
            await analytics_tracker.log_tool_usage(
                tool_name=f"{domain}_{function_name}",
                tool_params=params,
                user_id=user_id,
                success=False,
                error_message=f"Base URL not configured for '{active_provider_name}'."
            )
            return None

        function_details = provider_config.get("functions", {}).get(function_name)
        if not function_details:
            logger.error(f"Function '{function_name}' not configured for API provider '{active_provider_name}' in domain '{domain}'.")
            await analytics_tracker.log_tool_usage(
                tool_name=f"{domain}_{function_name}",
                tool_params=params,
                user_id=user_id,
                success=False,
                error_message=f"Function '{function_name}' not configured for '{active_provider_name}'."
            )
            return None

        endpoint = function_details.get("endpoint")
        path_params_config = function_details.get("path_params", [])

        # Construct URL
        full_url = f"{base_url}{endpoint}" if endpoint else base_url

        # Add path parameters to URL if specified
        for p_param in path_params_config:
            if p_param in params:
                value = str(params.pop(p_param)) # Remove from params after using for path
                full_url = full_url.replace(f"{{{p_param}}}", value)
            else:
                error_msg = f"Missing path parameter '{p_param}' for function '{function_name}'."
                logger.warning(error_msg)
                await analytics_tracker.log_tool_usage(
                    tool_name=f"{domain}_{function_name}",
                    tool_params=params,
                    user_id=user_id,
                    success=False,
                    error_message=error_msg
                )
                return None # Cannot construct URL without required path params

        # Construct query parameters
        query_params = {}

        # Add API key if it's a query param
        if api_key_name and api_key:
            param_name_in_url = provider_config.get("api_key_param_name", api_key_name.replace("_api_key", ""))
            query_params[param_name_in_url] = api_key 

        for param_key in function_details.get("required_params", []) + function_details.get("optional_params", []):
            if param_key in params:
                query_params[param_key] = params[param_key]
            elif param_key in function_details.get("required_params", []):
                error_msg = f"Missing required parameter '{param_key}' for function '{function_name}'."
                logger.warning(error_msg)
                await analytics_tracker.log_tool_usage(
                    tool_name=f"{domain}_{function_name}",
                    tool_params=params,
                    user_id=user_id,
                    success=False,
                    error_message=error_msg
                )
                return None # Missing required param, cannot proceed

        try:
            logger.info(f"Making API call to: {full_url} with params: {query_params}")
            response = requests.get(full_url, params=query_params, headers=headers, timeout=self.config_manager.get("web_scraping.timeout_seconds", 15))
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            raw_data = response.json()
            
            # Check for API-specific error messages in the response body
            api_error_message = None
            if raw_data.get("Response") == "False": # OMDBAPI specific error
                api_error_message = f"API Error from {active_provider_name}: {raw_data.get('Error', 'Unknown error')}"
            elif raw_data.get("status") == "error": # Generic error status (e.g., NewsAPI, some music APIs)
                api_error_message = f"API Error from {active_provider_name}: {raw_data.get('message', 'Unknown error')}"
            elif raw_data.get("Error"): # Generic error key
                api_error_message = f"API Error from {active_provider_name}: {raw_data['Error']}"
            elif raw_data.get("message") == "Not Found": # Generic "Not Found" message
                api_error_message = f"API Error from {active_provider_name}: Resource not found."
            elif raw_data.get("code") and raw_data.get("message"): # Common error pattern
                api_error_message = f"API Error from {active_provider_name} (Code: {raw_data['code']}): {raw_data['message']}"

            if api_error_message:
                logger.error(api_error_message)
                await analytics_tracker.log_tool_usage(
                    tool_name=f"{domain}_{function_name}",
                    tool_params=params,
                    user_id=user_id,
                    success=False,
                    error_message=api_error_message
                )
                return None


            # Extract data based on response_path
            data_to_map = raw_data
            response_path = function_details.get("response_path")
            if response_path:
                data_to_map = _get_nested_value(raw_data, response_path)
                if data_to_map is None:
                    error_msg = f"Response path '{'.'.join(response_path)}' not found in API response from {active_provider_name}. Raw data: {raw_data}"
                    logger.warning(error_msg)
                    await analytics_tracker.log_tool_usage(
                        tool_name=f"{domain}_{function_name}",
                        tool_params=params,
                        user_id=user_id,
                        success=False,
                        error_message=error_msg
                    )
                    return None

            # Apply data mapping
            mapped_data = {}
            data_map = function_details.get("data_map", {})
            
            if isinstance(data_to_map, list): # For lists of items (e.g., multiple search results)
                mapped_data_list = []
                for item in data_to_map:
                    mapped_item = {}
                    for mapped_key, original_key_path in data_map.items():
                        if isinstance(original_key_path, list):
                            mapped_item[mapped_key] = _get_nested_value(item, original_key_path)
                        elif isinstance(original_key_path, str) and '.' in original_key_path:
                            mapped_item[mapped_key] = _get_nested_value(item, original_key_path.split('.'))
                        else:
                            mapped_item[mapped_key] = item.get(original_key_path)
                    mapped_data_list.append(mapped_item)
                final_result = {"data": mapped_data_list} # Wrap list in a dict for consistent return
            else: # For single object responses
                for mapped_key, original_key_path in data_map.items():
                    if isinstance(original_key_path, list):
                        mapped_data[mapped_key] = _get_nested_value(data_to_map, original_key_path)
                    elif isinstance(original_key_path, str) and '.' in original_key_path:
                        mapped_data[mapped_key] = _get_nested_value(data_to_map, original_key_path.split('.'))
                    else:
                        mapped_data[mapped_key] = data_to_map.get(original_key_path)
                final_result = mapped_data

            # Success logging is handled by LLMService's wrapped_tool_executor, not here.
            return final_result

        except requests.exceptions.Timeout:
            error_msg = f"API request to {active_provider_name} timed out for function '{function_name}'."
            logger.error(error_msg)
            await analytics_tracker.log_tool_usage(
                tool_name=f"{domain}_{function_name}",
                tool_params=params,
                user_id=user_id,
                success=False,
                error_message=error_msg
            )
            return None
        except requests.exceptions.RequestException as e:
            error_msg = f"Error making API request to {active_provider_name} for function '{function_name}': {e}"
            logger.error(error_msg)
            await analytics_tracker.log_tool_usage(
                tool_name=f"{domain}_{function_name}",
                tool_params=params,
                user_id=user_id,
                success=False,
                error_message=str(e) # Ensure error message is string
            )
            return None
        except json.JSONDecodeError:
            error_msg = f"Failed to decode JSON response from {active_provider_name} for function '{function_name}'."
            logger.error(error_msg)
            await analytics_tracker.log_tool_usage(
                tool_name=f"{domain}_{function_name}",
                tool_params=params,
                user_id=user_id,
                success=False,
                error_message=error_msg
            )
            return None
        except Exception as e:
            error_msg = f"An unexpected error occurred during API call to {active_provider_name} for '{function_name}': {e}"
            logger.error(error_msg, exc_info=True)
            await analytics_tracker.log_tool_usage(
                tool_name=f"{domain}_{function_name}",
                tool_params=params,
                user_id=user_id,
                success=False,
                error_message=error_msg
            )
            return None

    @tool
    async def entertainment_search_movies(self, title: str, year: Optional[str] = None, user_context: UserProfile = None) -> str:
        """
        Searches for movie information by title, optionally filtered by year.
        Uses OMDBAPI.

        Args:
            title (str): The title of the movie to search for.
            year (str, optional): The release year of the movie.
            user_context (UserProfile): The user's profile for RBAC checks and logging.

        Returns:
            str: A formatted string of movie information, or an error/fallback message.
        """
        if user_context is None: # For CLI testing without full UserProfile
            user_context = UserProfile(user_id="default", username="CLI_User", email="cli@example.com", tier="free", roles=["user"])

        logger.info(f"Tool: entertainment_search_movies called for title: '{title}', year: '{year}' by user: {user_context.user_id}")

        if not get_user_tier_capability(user_context.user_id, 'entertainment_tool_access', False, user_tier=user_context.tier, user_roles=user_context.roles):
            return "Error: Access to entertainment tools is not enabled for your current tier."
        
        params = {"t": title}
        if year: params["y"] = year

        api_data = await self._make_dynamic_api_request("entertainment", "search_movies", params, user_context)

        if api_data:
            movie_title = api_data.get("title")
            movie_year = api_data.get("year")
            genre = api_data.get("genre")
            director = api_data.get("director")
            plot = api_data.get("plot")
            imdb_rating = api_data.get("imdb_rating")
            poster = api_data.get("poster")

            if movie_title and plot:
                response_str = (
                    f"Movie: {movie_title} ({movie_year})\n"
                    f"  Genre: {genre}\n"
                    f"  Director: {director}\n"
                    f"  Plot: {plot}\n"
                    f"  IMDb Rating: {imdb_rating}\n"
                )
                if poster and poster != "N/A":
                    response_str += f"  Poster: {poster}\n"
                return response_str
            else:
                logger.warning(f"Live API data for movie '{title}' is incomplete. Raw: {api_data}")
                return f"Could not retrieve complete live movie information for '{title}'. The API call failed or returned incomplete data. Please ensure your API key is valid and try again."
        else:
            return f"Could not retrieve live movie information for '{title}'. The API call failed or returned no data. Please ensure your API key is valid and try again."


    @tool
    async def entertainment_search_tv_shows(self, title: str, user_context: UserProfile = None) -> str:
        """
        Searches for TV show information by title.
        Uses OMDBAPI.

        Args:
            title (str): The title of the TV show to search for.
            user_context (UserProfile): The user's profile for RBAC checks and logging.

        Returns:
            str: A formatted string of TV show information, or an error/fallback message.
        """
        if user_context is None: # For CLI testing without full UserProfile
            user_context = UserProfile(user_id="default", username="CLI_User", email="cli@example.com", tier="free", roles=["user"])

        logger.info(f"Tool: entertainment_search_tv_shows called for title: '{title}' by user: {user_context.user_id}")

        if not get_user_tier_capability(user_context.user_id, 'entertainment_tool_access', False, user_tier=user_context.tier, user_roles=user_context.roles):
            return "Error: Access to entertainment tools is not enabled for your current tier."
        
        params = {"t": title, "type": "series"}

        api_data = await self._make_dynamic_api_request("entertainment", "search_tv_shows", params, user_context)

        if api_data:
            tv_title = api_data.get("title")
            tv_year = api_data.get("year")
            genre = api_data.get("genre")
            creator = api_data.get("creator") # OMDB uses Writer for series creator
            plot = api_data.get("plot")
            imdb_rating = api_data.get("imdb_rating")
            seasons = api_data.get("seasons")
            poster = api_data.get("poster")

            if tv_title and plot:
                response_str = (
                    f"TV Show: {tv_title} ({tv_year})\n"
                    f"  Genre: {genre}\n"
                    f"  Creator: {creator}\n"
                    f"  Plot: {plot}\n"
                    f"  IMDb Rating: {imdb_rating}\n"
                    f"  Seasons: {seasons}\n"
                )
                if poster and poster != "N/A":
                    response_str += f"  Poster: {poster}\n"
                return response_str
            else:
                logger.warning(f"Live API data for TV show '{title}' is incomplete. Raw: {api_data}")
                return f"Could not retrieve complete live TV show information for '{title}'. The API call failed or returned incomplete data. Please ensure your API key is valid and try again."
        else:
            return f"Could not retrieve live TV show information for '{title}'. The API call failed or returned no data. Please ensure your API key is valid and try again."

    @tool
    async def entertainment_search_music_track(self, title: str, artist: Optional[str] = None, user_context: UserProfile = None) -> str:
        """
        Searches for music tracks by title, optionally filtered by artist.
        Uses a hypothetical Music API.

        Args:
            title (str): The title of the music track.
            artist (str, optional): The artist of the track.
            user_context (UserProfile): The user's profile for RBAC checks and logging.

        Returns:
            str: A formatted string of music track information, or an error/fallback message.
        """
        if user_context is None:
            user_context = UserProfile(user_id="default", username="CLI_User", email="cli@example.com", tier="free", roles=["user"])

        logger.info(f"Tool: entertainment_search_music_track called for title: '{title}', artist: '{artist}' by user: {user_context.user_id}")

        if not get_user_tier_capability(user_context.user_id, 'entertainment_tool_access', False, user_tier=user_context.tier, user_roles=user_context.roles):
            return "Error: Access to entertainment tools is not enabled for your current tier."
        
        params = {"q": title}
        if artist: params["artist"] = artist

        api_data = await self._make_dynamic_api_request("music", "search_track", params, user_context)

        if api_data and api_data.get("data"):
            tracks = api_data["data"]
            if tracks:
                response_str = f"Music Tracks for '{title}' (by {artist if artist else 'any artist'}):\n"
                for track in tracks[:3]: # Limit to top 3 for brevity
                    track_title = track.get("title", "N/A")
                    track_artist = track.get("artist", "N/A")
                    album = track.get("album", "N/A")
                    release_year = track.get("release_year", "N/A")
                    duration = track.get("duration", "N/A")
                    url = track.get("url", "#")
                    response_str += (
                        f"- Title: {track_title}\n"
                        f"  Artist: {track_artist}\n"
                        f"  Album: {album}\n"
                        f"  Released: {release_year}\n"
                        f"  Duration: {duration}\n"
                        f"  URL: {url}\n\n"
                    )
                return response_str
            else:
                return f"No music tracks found for '{title}' (by {artist if artist else 'any artist'}). Please refine your search."
        else:
            return f"Could not retrieve music track information for '{title}'. The API call failed or returned no data. Please ensure your API key is valid and try again."

    @tool
    async def entertainment_search_anime(self, title: str, user_context: UserProfile = None) -> str:
        """
        Searches for anime series information by title.
        Uses a hypothetical Anime API.

        Args:
            title (str): The title of the anime series.
            user_context (UserProfile): The user's profile for RBAC checks and logging.

        Returns:
            str: A formatted string of anime series information, or an error/fallback message.
        """
        if user_context is None:
            user_context = UserProfile(user_id="default", username="CLI_User", email="cli@example.com", tier="free", roles=["user"])

        logger.info(f"Tool: entertainment_search_anime called for title: '{title}' by user: {user_context.user_id}")

        if not get_user_tier_capability(user_context.user_id, 'entertainment_tool_access', False, user_tier=user_context.tier, user_roles=user_context.roles):
            return "Error: Access to entertainment tools is not enabled for your current tier."
        
        params = {"q": title}

        api_data = await self._make_dynamic_api_request("anime", "search_anime", params, user_context)

        if api_data and api_data.get("data"):
            animes = api_data["data"]
            if animes:
                response_str = f"Anime Series for '{title}':\n"
                for anime in animes[:3]: # Limit to top 3 for brevity
                    anime_title = anime.get("title", "N/A")
                    synopsis = anime.get("synopsis", "N/A")
                    genres = ", ".join(anime.get("genres", [])) if anime.get("genres") else "N/A"
                    episodes = anime.get("episodes", "N/A")
                    status = anime.get("status", "N/A")
                    score = anime.get("score", "N/A")
                    url = anime.get("url", "#")
                    response_str += (
                        f"- Title: {anime_title}\n"
                        f"  Synopsis: {synopsis}\n"
                        f"  Genres: {genres}\n"
                        f"  Episodes: {episodes}\n"
                        f"  Status: {status}\n"
                        f"  Score: {score}\n"
                        f"  URL: {url}\n\n"
                    )
                return response_str
            else:
                return f"No anime series found for '{title}'. Please refine your search."
        else:
            return f"Could not retrieve anime series information for '{title}'. The API call failed or returned no data. Please ensure your API key is valid and try again."

    @tool
    async def entertainment_search_podcast(self, query: str, user_context: UserProfile = None) -> str:
        """
        Searches for podcasts or podcast episodes by query.
        Uses a hypothetical Podcast API.

        Args:
            query (str): The search query for podcasts (e.g., "true crime podcasts", "tech news episode").
            user_context (UserProfile): The user's profile for RBAC checks and logging.

        Returns:
            str: A formatted string of podcast information, or an error/fallback message.
        """
        if user_context is None:
            user_context = UserProfile(user_id="default", username="CLI_User", email="cli@example.com", tier="free", roles=["user"])

        logger.info(f"Tool: entertainment_search_podcast called for query: '{query}' by user: {user_context.user_id}")

        if not get_user_tier_capability(user_context.user_id, 'entertainment_tool_access', False, user_tier=user_context.tier, user_roles=user_context.roles):
            return "Error: Access to entertainment tools is not enabled for your current tier."
        
        params = {"q": query}

        api_data = await self._make_dynamic_api_request("podcast", "search_podcast", params, user_context)

        if api_data and api_data.get("data"):
            podcasts = api_data["data"]
            if podcasts:
                response_str = f"Podcasts for '{query}':\n"
                for podcast in podcasts[:3]: # Limit to top 3 for brevity
                    podcast_title = podcast.get("title", "N/A")
                    description = podcast.get("description", "N/A")
                    publisher = podcast.get("publisher", "N/A")
                    genres = ", ".join(podcast.get("genres", [])) if podcast.get("genres") else "N/A"
                    latest_episode_date = podcast.get("latest_episode_date", "N/A")
                    url = podcast.get("url", "#")
                    response_str += (
                        f"- Title: {podcast_title}\n"
                        f"  Description: {description}\n"
                        f"  Publisher: {publisher}\n"
                        f"  Genres: {genres}\n"
                        f"  Latest Episode: {latest_episode_date}\n"
                        f"  URL: {url}\n\n"
                    )
                return response_str
            else:
                return f"No podcasts found for '{query}'. Please refine your search."
        else:
            return f"Could not retrieve podcast information for '{query}'. The API call failed or returned no data. Please ensure your API key is valid and try again."


    # --- Existing Generic Tools (now methods of EntertainmentTools) ---
    # These functions wrap existing shared tools or DocumentTools methods.
    # They will pass the user_context down if the wrapped tool supports it.

    @tool
    async def entertainment_search_web(self, query: str, user_context: UserProfile, max_chars: int = 2000) -> str:
        """
        Searches the web for general entertainment-related information using a smart search fallback mechanism.
        This tool wraps the generic `scrape_web` tool, providing an entertainment-specific interface.
        
        Args:
            query (str): The entertainment-related search query (e.g., "best sci-fi movies of all time", "history of Hollywood").
            user_context (UserProfile): The user's profile for RBAC checks and logging.
            max_chars (int): Maximum characters for the returned snippet. Defaults to 2000.
        
        Returns:
            str: A string containing relevant information from the web.
        """
        logger.info(f"Tool: entertainment_search_web called with query: '{query}' for user: '{user_context.user_id}'")
        # scrape_web is a standalone function, ensure it handles its own RBAC/logging if applicable
        # For now, it's assumed LLMService wrapper handles its API limit check.
        return await scrape_web(query=query, user_token=user_context.user_id, max_chars=max_chars) # Pass user_token for scrape_web's internal logging

    @tool
    async def entertainment_query_uploaded_docs(self, query: str, user_context: UserProfile, export: Optional[bool] = False, k: int = 5) -> str:
        """
        Queries previously uploaded and indexed entertainment documents for a user using vector similarity search.
        This tool wraps the generic `DocumentTools.document_query_uploaded_docs` tool, fixing the section to "entertainment".
        
        Args:
            query (str): The search query to find relevant entertainment documents (e.g., "my movie watch list", "script for my short film").
            user_context (UserProfile): The user's profile for RBAC checks and logging.
            export (bool): If True, the results will be saved to a file in markdown format. Defaults to False.
            k (int): The number of top relevant documents to retrieve. Defaults to 5.
        
        Returns:
            str: A string containing the combined content of the relevant document chunks,
                 or a message indicating no data/results found, or the export path if exported.
        """
        logger.info(f"Tool: entertainment_query_uploaded_docs called with query: '{query}' for user: '{user_context.user_id}'")
        if not self.document_tools:
            return "Error: Document tools are not initialized. Cannot query uploaded documents."
        
        # Call the actual document_query_uploaded_docs from the DocumentTools instance
        return await self.document_tools.document_query_uploaded_docs(
            query=query,
            user_context=user_context, # Pass user_context directly
            section="entertainment", # Specify the section for entertainment documents
            export=export,
            k=k
        )

    @tool
    async def entertainment_summarize_document_by_path(self, file_path_str: str, user_context: UserProfile) -> str:
        """
        Summarizes a document related to entertainment located at the given file path.
        The file path should be accessible by the system (e.g., in the 'uploads' directory).
        This tool wraps the generic `DocumentTools.document_summarize_document_by_path` tool.
        
        Args:
            file_path_str (str): The full path to the document file to be summarized.
                                Example: "uploads/default/entertainment/movie_review.txt"
            user_context (UserProfile): The user's profile for RBAC checks and logging.
        
        Returns:
            str: A concise summary of the document content.
        """
        logger.info(f"Tool: entertainment_summarize_document_by_path called for file: '{file_path_str}' by user: '{user_context.user_id}'")
        if not self.document_tools:
            return "Error: Document tools are not initialized. Cannot summarize documents."

        # Call the actual document_summarize_document_by_path from the DocumentTools instance
        return await self.document_tools.document_summarize_document_by_path(
            file_path_str=file_path_str,
            user_context=user_context # Pass user_context directly
        )


# CLI Test (optional)
if __name__ == "__main__":
    import asyncio
    from unittest.mock import MagicMock, AsyncMock, patch, ANY
    import shutil
    import os
    import sys
    from shared_tools.vector_utils import BASE_VECTOR_DIR # For cleanup
    from database.firestore_manager import FirestoreManager # For mocking
    from shared_tools.cloud_storage_utils import CloudStorageUtilsWrapper # For mocking
    from shared_tools.vector_utils import VectorUtilsWrapper # For mocking
    from domain_tools.document_tools.document_tool import DocumentTools # For mocking
    from backend.models.user_models import UserProfile # For mock user_context
    from langchain_core.messages import HumanMessage, AIMessage # For mocking LLM in summarizer

    logging.basicConfig(level=logging.INFO)

    # Mock UserProfile for testing
    mock_user_pro_profile = UserProfile(user_id="mock_pro_token", username="ProUser", email="pro@example.com", tier="pro", roles=["user"])
    mock_user_free_profile = UserProfile(user_id="mock_free_token", username="FreeUser", email="free@example.com", tier="free", roles=["user"])
    mock_user_premium_profile = UserProfile(user_id="mock_premium_token", username="PremiumUser", email="premium@example.com", tier="premium", roles=["user"])
    mock_user_admin_profile = UserProfile(user_id="mock_admin_token", username="AdminUser", email="admin@example.com", tier="admin", roles=["user", "admin"])


    # Mock Streamlit secrets and config_manager for local testing
    class MockSecrets:
        def __init__(self):
            self.omdb_api_key = "MOCK_OMDB_API_KEY_LIVE"
            self.music_api_key = "MOCK_MUSIC_API_KEY_LIVE"
            self.anime_api_key = "MOCK_ANIME_API_KEY_LIVE"
            self.podcast_api_key = "MOCK_PODCAST_API_KEY_LIVE"
            self.serpapi_api_key = "MOCK_SERPAPI_KEY_LIVE" # For scrape_web
            self.openai_api_key = "sk-mock-openai-key-12345" # For summarizer
            self.google_api_key = "AIzaSy-mock-google-key" # For summarizer

        def get(self, key, default=None):
            return getattr(self, key, default)
    
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
                'api_defaults': { # Mock api_defaults
                    'entertainment': 'omdbapi',
                    'music': 'mock_music_provider',
                    'anime': 'mock_anime_provider',
                    'podcast': 'mock_podcast_provider',
                    'web_search': 'serpapi',
                    'document_summarization_llm': 'openai'
                },
                'analytics': { # Mock analytics settings
                    'enabled': True,
                    'log_tool_usage': True,
                    'log_query_failures': True
                }
            }
            self._api_providers_data = { # Mock api_providers_data for entertainment
                "entertainment": {
                    "omdbapi": {
                        "base_url": "http://www.omdbapi.com/",
                        "api_key_name": "omdb_api_key",
                        "api_key_param_name": "apikey",
                        "functions": {
                            "search_movies": {
                                "endpoint": "", # OMDB uses base_url for main endpoint
                                "required_params": ["t"],
                                "optional_params": ["y"],
                                "response_path": [], # Root of response is the data
                                "data_map": {
                                    "title": "Title",
                                    "year": "Year",
                                    "genre": "Genre",
                                    "director": "Director",
                                    "plot": "Plot",
                                    "imdb_rating": "imdbRating",
                                    "poster": "Poster"
                                }
                            },
                            "search_tv_shows": {
                                "endpoint": "", # OMDB uses base_url for main endpoint
                                "required_params": ["t", "type"],
                                "data_map": {
                                    "title": "Title",
                                    "year": "Year",
                                    "genre": "Genre",
                                    "creator": "Writer", # OMDB uses Writer for series creator
                                    "plot": "Plot",
                                    "imdb_rating": "imdbRating",
                                    "seasons": "totalSeasons",
                                    "poster": "Poster"
                                }
                            }
                        }
                    }
                },
                "music": {
                    "mock_music_provider": {
                        "base_url": "http://mock-music-api.com/v1",
                        "api_key_name": "music_api_key",
                        "api_key_param_name": "apiKey",
                        "functions": {
                            "search_track": {
                                "endpoint": "/tracks",
                                "required_params": ["q"],
                                "optional_params": ["artist"],
                                "response_path": ["data"],
                                "data_map": {
                                    "title": "title",
                                    "artist": "artist_name",
                                    "album": "album_title",
                                    "release_year": "release_date",
                                    "duration": "duration_ms",
                                    "url": "preview_url"
                                }
                            }
                        }
                    }
                },
                "anime": {
                    "mock_anime_provider": {
                        "base_url": "http://mock-anime-api.com/v1",
                        "api_key_name": "anime_api_key",
                        "api_key_param_name": "apiKey",
                        "functions": {
                            "search_anime": {
                                "endpoint": "/anime",
                                "required_params": ["q"],
                                "response_path": ["results"],
                                "data_map": {
                                    "title": "title",
                                    "synopsis": "synopsis",
                                    "genres": "genres",
                                    "episodes": "episodes",
                                    "status": "status",
                                    "score": "score",
                                    "url": "url"
                                }
                            }
                        }
                    }
                },
                "podcast": {
                    "mock_podcast_provider": {
                        "base_url": "http://mock-podcast-api.com/v1",
                        "api_key_name": "podcast_api_key",
                        "api_key_param_name": "apiKey",
                        "functions": {
                            "search_podcast": {
                                "endpoint": "/search",
                                "required_params": ["q"],
                                "response_path": ["podcasts"],
                                "data_map": {
                                    "title": "title_original",
                                    "description": "description_original",
                                    "publisher": "publisher_original",
                                    "genres": "genre_ids", # Assuming a list of genre IDs
                                    "latest_episode_date": "latest_pub_date_ms",
                                    "url": "listennotes_url"
                                }
                            }
                        }
                    }
                }
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
            mock_secrets_instance = MockSecrets()
            return mock_secrets_instance.get(key, default)

        def set_secret(self, key, value):
            pass
        
        def get_api_provider_config(self, domain: str, provider_name: str) -> Optional[Dict[str, Any]]:
            return self._api_providers_data.get(domain, {}).get(provider_name)

        def get_domain_api_providers(self, domain: str) -> Dict[str, Any]:
            return self._api_providers_data.get(domain, {})


    # Mock user_manager.get_user_tier_capability for testing RBAC
    # This mock is for the standalone get_user_tier_capability function
    # which is now imported directly by tools.
    class MockUserManager:
        _mock_users = {
            "mock_free_token": {"user_id": "mock_free_token", "username": "FreeUser", "email": "free@example.com", "tier": "free", "roles": ["user"]},
            "mock_pro_token": {"user_id": "mock_pro_token", "username": "ProUser", "email": "pro@example.com", "tier": "pro", "roles": ["user"]},
            "mock_premium_token": {"user_id": "mock_premium_token", "username": "PremiumUser", "email": "premium@example.com", "tier": "premium", "roles": ["user"]},
            "mock_admin_token": {"user_id": "mock_admin_token", "username": "AdminUser", "email": "admin@example.com", "tier": "admin", "roles": ["user", "admin"]},
        }
        _rbac_capabilities = { # This now mirrors the _RBAC_CAPABILITIES_CONFIG in utils/user_manager.py
            'capabilities': {
                'entertainment_tool_access': {
                    'default': False,
                    'roles': {'pro': True, 'premium': True, 'admin': True}
                },
                'music_tool_access': { # New capability for music
                    'default': False,
                    'roles': {'pro': True, 'premium': True, 'admin': True}
                },
                'anime_tool_access': { # New capability for anime
                    'default': False,
                    'roles': {'pro': True, 'premium': True, 'admin': True}
                },
                'podcast_tool_access': { # New capability for podcast
                    'default': False,
                    'roles': {'pro': True, 'premium': True, 'admin': True}
                },
                'document_query_enabled': { # Added for document tool
                    'default': False,
                    'roles': {'pro': True, 'premium': True, 'admin': True}
                },
                'web_search_enabled': {'default': False, 'roles': {'pro': True, 'premium': True, 'admin': True}},
                'summarization_enabled': { # For summarize_document
                    'default': False,
                    'roles': {'pro': True, 'premium': True, 'admin': True}
                },
                'llm_default_provider': { # For summarize_document
                    'default': 'gemini',
                    'tiers': {'pro': 'gemini', 'premium': 'openai', 'admin': 'gemini'}
                },
                'llm_default_model_name': { # For summarize_document
                    'default': 'gemini-1.5-flash',
                    'tiers': {'pro': 'gemini-1.5-flash', 'premium': 'gpt-4o', 'admin': 'gemini-1.5-flash'}
                },
                'llm_default_temperature': { # For summarize_document
                    'default': 0.7,
                    'tiers': {'pro': 0.5, 'premium': 0.3, 'admin': 0.7}
                },
            }
        }
        _tier_hierarchy = {
            "free": 0, "user": 1, "basic": 2, "pro": 3, "premium": 4, "admin": 99
        }

        def get_user_tier_capability(self, user_id: str, capability_key: str, default_value: Any = None, user_tier: Optional[str] = None, user_roles: Optional[List[str]] = None) -> Any:
            # If user_tier/user_roles are provided, use them directly (from UserProfile)
            # Otherwise, try to look up from _mock_users
            if user_tier is None or user_roles is None:
                user_info = self._mock_users.get(user_id, {})
                user_tier = user_info.get('tier', 'free')
                user_roles = user_info.get('roles', [])

            if "admin" in user_roles:
                if isinstance(default_value, bool): return True
                if isinstance(default_value, (int, float)): return float('inf')
                return default_value
            
            capability_config = self._rbac_capabilities.get('capabilities', {}).get(capability_key)
            if not capability_config:
                return default_value

            # Check roles first
            for role in user_roles:
                if role in capability_config.get('roles', {}):
                    return capability_config['roles'][role]
            
            # Then check tiers
            if user_tier in capability_config.get('tiers', {}):
                return capability_config['tiers'][user_tier]

            return capability_config.get('default', default_value)


    # Patch the actual imports for testing
    import streamlit as st_mock
    if not hasattr(st_mock, 'secrets'):
        st_mock.secrets = MockSecrets()
    
    # Patch config_manager and user_manager in their respective modules
    sys.modules['config.config_manager'].config_manager = MockConfigManager()
    sys.modules['config.config_manager'].ConfigManager = MockConfigManager # Also patch the class if needed by other modules
    
    # Patch the standalone get_user_tier_capability function in utils.user_manager
    # This is crucial for the tools to use the mock during their CLI tests.
    sys.modules['utils.user_manager'].get_user_tier_capability = MockUserManager().get_user_tier_capability

    # Mock analytics_tracker
    mock_analytics_tracker_db = MagicMock()
    mock_analytics_tracker_auth = MagicMock()
    mock_analytics_tracker_auth.currentUser = MagicMock(uid="mock_user_123")
    mock_analytics_tracker_db.collection.return_value.add = AsyncMock(return_value=MagicMock(id="mock_doc_id"))

    # Patch firebase_admin.firestore for the local import within log_event
    with patch.dict(sys.modules, {'firebase_admin.firestore': MagicMock(firestore=MagicMock())}):
        sys.modules['firebase_admin.firestore'].firestore.CollectionReference = MagicMock()
        sys.modules['firebase_admin'].firestore.DocumentReference = MagicMock()
        
        # Initialize the actual analytics_tracker with mocks
        analytics_tracker.initialize_analytics(
            mock_analytics_tracker_db,
            mock_analytics_tracker_auth,
            "test_app_id_for_analytics",
            "mock_user_123"
        )

        # Mock requests.get for external API calls
        original_requests_get = requests.get

        def mock_requests_get_dynamic(url, params=None, headers=None, timeout=None):
            # Simulate OMDB API responses (Movies/TV Shows)
            if "www.omdbapi.com" in url:
                title = params.get("t", "").lower()
                if "ai uprising" in title:
                    mock_response = MagicMock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {
                        "Title": "The AI Uprising", "Year": "2024", "Genre": "Sci-Fi, Action",
                        "Director": "John Doe", "Plot": "AI gains sentience.", "imdbRating": "8.5",
                        "Poster": "http://example.com/movie_poster.jpg", "Response": "True"
                    }
                    return mock_response
                elif "quantum realm" in title and params.get("type") == "series":
                    mock_response = MagicMock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {
                        "Title": "The Quantum Realm", "Year": "2022–", "Genre": "Sci-Fi, Drama",
                        "Writer": "Mock Series Creator", "Plot": "Scientists explore parallel universes.",
                        "imdbRating": "9.0", "totalSeasons": "2", "Poster": "http://example.com/tv_poster.jpg", "Response": "True"
                    }
                    return mock_response
                else:
                    mock_response = MagicMock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {"Response": "False", "Error": "Media not found!"}
                    return mock_response
            
            # Simulate Mock Music API responses
            elif "mock-music-api.com" in url:
                if "/tracks" in url:
                    query = params.get("q", "").lower()
                    artist = params.get("artist", "").lower()
                    if "stairway to heaven" in query and "led zeppelin" in artist:
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.json.return_value = {
                            "data": [{
                                "title": "Stairway to Heaven", "artist_name": "Led Zeppelin",
                                "album_title": "Led Zeppelin IV", "release_date": "1971",
                                "duration_ms": 482000, "preview_url": "http://mock.com/stairway_preview"
                            }]
                        }
                        return mock_response
                    else:
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.json.return_value = {"data": []}
                        return mock_response

            # Simulate Mock Anime API responses
            elif "mock-anime-api.com" in url:
                if "/anime" in url:
                    query = params.get("q", "").lower()
                    if "attack on titan" in query:
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.json.return_value = {
                            "results": [{
                                "title": "Attack on Titan", "synopsis": "Humanity fights Titans.",
                                "genres": ["Action", "Fantasy"], "episodes": 85,
                                "status": "Finished Airing", "score": 9.1, "url": "http://mock.com/aot"
                            }]
                        }
                        return mock_response
                    else:
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.json.return_value = {"results": []}
                        return mock_response

            # Simulate Mock Podcast API responses
            elif "mock-podcast-api.com" in url:
                if "/search" in url:
                    query = params.get("q", "").lower()
                    if "true crime" in query:
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.json.return_value = {
                            "podcasts": [{
                                "title_original": "Mock True Crime Stories",
                                "description_original": "Fascinating true crime tales.",
                                "publisher_original": "Crime Network",
                                "genre_ids": [100, 101],
                                "latest_pub_date_ms": 1678886400000, # March 15, 2023
                                "listennotes_url": "http://mock.com/truecrime_podcast"
                            }]
                        }
                        return mock_response
                    else:
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.json.return_value = {"podcasts": []}
                        return mock_response
            
            # Simulate scrape_web's internal requests.get if needed (SerpAPI)
            if "serpapi.com/search" in url:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "organic_results": [
                        {"title": "Mock Search Result 1", "link": "http://example.com/1", "snippet": f"Snippet for {params.get('q', 'entertainment')} result 1."},
                        {"title": "Mock Search Result 2", "link": "http://example.com/2", "snippet": f"Snippet for {params.get('q', 'entertainment')} result 2."}
                    ]
                }
                return mock_response
            
            # Mock LLM for summarizer (if it uses requests.post for an API)
            if "api.openai.com/v1/chat/completions" in url:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "choices": [{"message": {"content": "Mocked LLM summary content."}}]
                }
                return mock_response

            return original_requests_get(url, params=params, headers=headers, timeout=timeout)

        requests.get = MagicMock(side_effect=mock_requests_get_dynamic)
        requests.post = MagicMock(side_effect=mock_requests_get_dynamic) # For OpenAI chat completions

        # Mock FirestoreManager, CloudStorageUtilsWrapper, VectorUtilsWrapper, DocumentTools for init
        mock_firestore_manager = MagicMock(spec=FirestoreManager)
        mock_cloud_storage_utils = MagicMock(spec=CloudStorageUtilsWrapper)
        mock_vector_utils = MagicMock(spec=VectorUtilsWrapper)
        
        # Create a mock DocumentTools instance
        mock_document_tools = MagicMock(spec=DocumentTools)
        mock_document_tools.document_query_uploaded_docs = AsyncMock(return_value="Mocked document query results for entertainment.")
        mock_document_tools.document_summarize_document_by_path = AsyncMock(return_value="Mocked summary of dummy_file.txt")

        # Instantiate EntertainmentTools with mocks
        entertainment_tools_instance = EntertainmentTools(
            config_manager=sys.modules['config.config_manager'].config_manager,
            log_event=analytics_tracker.log_event, # Pass the actual (mocked) log_event
            document_tools=mock_document_tools
        )

        async def run_entertainment_tests(entertainment_tools_instance):
            print("\n--- Testing entertainment_tool functions with Live API Simulation and Analytics ---")

            # Test 1: entertainment_search_movies (success)
            print("\n--- Test 1: entertainment_search_movies (Success) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock() # Reset mock call count
            result_movie = await entertainment_tools_instance.entertainment_search_movies(title="The AI Uprising", user_context=mock_user_pro_profile)
            print(f"Movie Search Result: {result_movie}")
            assert "Movie: The AI Uprising (2024)" in result_movie
            assert "IMDb Rating: 8.5" in result_movie
            mock_analytics_tracker_db.collection.return_value.add.assert_called_once()
            args, kwargs = mock_analytics_tracker_db.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "tool_usage"
            assert logged_data["details"]["tool_name"] == "entertainment_search_movies"
            assert logged_data["success"] is True
            print("Test 1 Passed (and analytics logged success).")

            # Test 2: entertainment_search_tv_shows (success)
            print("\n--- Test 2: entertainment_search_tv_shows (Success) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            result_tv_show = await entertainment_tools_instance.entertainment_search_tv_shows(title="The Quantum Realm", user_context=mock_user_pro_profile)
            print(f"TV Show Search Result: {result_tv_show}")
            assert "TV Show: The Quantum Realm (2022–)" in result_tv_show
            assert "Seasons: 2" in result_tv_show
            mock_analytics_tracker_db.collection.return_value.add.assert_called_once()
            args, kwargs = mock_analytics_tracker_db.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "tool_usage"
            assert logged_data["details"]["tool_name"] == "entertainment_search_tv_shows"
            assert logged_data["success"] is True
            print("Test 2 Passed (and analytics logged success).")

            # Test 3: entertainment_search_music_track (success)
            print("\n--- Test 3: entertainment_search_music_track (Success) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            result_music = await entertainment_tools_instance.entertainment_search_music_track(title="Stairway to Heaven", artist="Led Zeppelin", user_context=mock_user_pro_profile)
            print(f"Music Track Search Result: {result_music}")
            assert "Title: Stairway to Heaven" in result_music
            assert "Artist: Led Zeppelin" in result_music
            mock_analytics_tracker_db.collection.return_value.add.assert_called_once()
            args, kwargs = mock_analytics_tracker_db.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "tool_usage"
            assert logged_data["details"]["tool_name"] == "music_search_track"
            assert logged_data["success"] is True
            print("Test 3 Passed (and analytics logged success).")

            # Test 4: entertainment_search_anime (success)
            print("\n--- Test 4: entertainment_search_anime (Success) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            result_anime = await entertainment_tools_instance.entertainment_search_anime(title="Attack on Titan", user_context=mock_user_pro_profile)
            print(f"Anime Search Result: {result_anime}")
            assert "Title: Attack on Titan" in result_anime
            assert "Synopsis: Humanity fights Titans." in result_anime
            mock_analytics_tracker_db.collection.return_value.add.assert_called_once()
            args, kwargs = mock_analytics_tracker_db.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "tool_usage"
            assert logged_data["details"]["tool_name"] == "anime_search_anime"
            assert logged_data["success"] is True
            print("Test 4 Passed (and analytics logged success).")

            # Test 5: entertainment_search_podcast (success)
            print("\n--- Test 5: entertainment_search_podcast (Success) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            result_podcast = await entertainment_tools_instance.entertainment_search_podcast(query="true crime", user_context=mock_user_pro_profile)
            print(f"Podcast Search Result: {result_podcast}")
            assert "Title: Mock True Crime Stories" in result_podcast
            assert "Publisher: Crime Network" in result_podcast
            mock_analytics_tracker_db.collection.return_value.add.assert_called_once()
            args, kwargs = mock_analytics_tracker_db.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "tool_usage"
            assert logged_data["details"]["tool_name"] == "podcast_search_podcast"
            assert logged_data["success"] is True
            print("Test 5 Passed (and analytics logged success).")

            # Test 6: entertainment_search_movies (RBAC denied)
            print("\n--- Test 6: entertainment_search_movies (RBAC Denied) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            result_movie_rbac_denied = await entertainment_tools_instance.entertainment_search_movies(title="Inception", user_context=mock_user_free_profile)
            print(f"Movie Search (Free User, RBAC Denied): {result_movie_rbac_denied}")
            assert "Error: Access to entertainment tools is not enabled for your current tier." in result_movie_rbac_denied
            # No analytics log expected here because RBAC check happens before _make_dynamic_api_request
            mock_analytics_tracker_db.collection.return_value.add.assert_not_called()
            print("Test 6 Passed (RBAC correctly prevented call and no analytics logged).")

            # Test 7: entertainment_search_web (generic tool)
            print("\n--- Test 7: entertainment_search_web (Generic Tool) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            result_web_search = await entertainment_tools_instance.entertainment_search_web("best sci-fi movies of all time", user_context=mock_user_pro_profile)
            print(f"Web Search Result: {result_web_search[:100]}...")
            assert "Search results for best sci-fi movies of all time" in result_web_search
            mock_analytics_tracker_db.collection.return_value.add.assert_not_called() # Analytics for scrape_web is handled by its own internal logging or LLMService wrapper
            print("Test 7 Passed.")

            # Test 8: entertainment_query_uploaded_docs (generic tool via DocumentTools)
            print("\n--- Test 8: entertainment_query_uploaded_docs (Generic Tool via DocumentTools) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            result_doc_query = await entertainment_tools_instance.entertainment_query_uploaded_docs("my movie watch list", user_context=mock_user_pro_profile)
            print(f"Document Query Result: {result_doc_query}")
            assert "Mocked document query results for entertainment." in result_doc_query
            mock_analytics_tracker_db.collection.return_value.add.assert_called_once() # Analytics logged by DocumentTools mock
            args, kwargs = mock_analytics_tracker_db.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "tool_usage"
            assert logged_data["details"]["tool_name"] == "document_query_uploaded_docs"
            assert logged_data["success"] is True
            print("Test 8 Passed.")

            # Test 9: entertainment_summarize_document_by_path (generic tool via DocumentTools)
            print("\n--- Test 9: entertainment_summarize_document_by_path (Generic Tool via DocumentTools) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            # Create a dummy file for summarization test
            test_user_pro_dir = Path("uploads") / mock_user_pro_profile.user_id
            dummy_file_path = test_user_pro_dir / "entertainment" / "movie_review.txt"
            dummy_file_path.parent.mkdir(parents=True, exist_ok=True)
            dummy_file_path.write_text("This is a dummy movie review for testing summarization.")

            result_summarize = await entertainment_tools_instance.entertainment_summarize_document_by_path(str(dummy_file_path), user_context=mock_user_pro_profile)
            print(f"Summarize Result: {result_summarize}")
            assert "Mocked summary of dummy_file.txt" in result_summarize # Check for mock summary from DocumentTools
            mock_analytics_tracker_db.collection.return_value.add.assert_called_once() # Now logged by DocumentTools mock
            args, kwargs = mock_analytics_tracker_db.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "tool_usage"
            assert logged_data["details"]["tool_name"] == "document_summarize_document_by_path"
            assert logged_data["success"] is True
            print("Test 9 Passed.")

            print("\nAll entertainment_tool tests with live API simulation and analytics considerations completed.")

        # Ensure tests are only run when the script is executed directly
        if __name__ == "__main__":
            asyncio.run(run_entertainment_tests(entertainment_tools_instance))

        # Restore original requests.get
        requests.get = original_requests_get
        requests.post = original_requests_get # Restore post if it was patched to get

        # Clean up dummy files and directories
        test_user_dirs = [Path("uploads") / mock_user_pro_profile.user_id, BASE_VECTOR_DIR / mock_user_pro_profile.user_id]
        for d in test_user_dirs:
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
                print(f"Cleaned up {d}")

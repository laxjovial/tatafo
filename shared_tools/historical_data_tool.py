import requests
import yaml

def get_api_config(provider_name: str) -> dict:
    """

def make_api_request(provider_name: str, function_name: str, params: dict, user_api_keys: list = []) -> dict:
    """
    Makes an API request to the given provider and function with the given parameters.

    :param provider_name: The name of the API provider.
    :param function_name: The name of the function to be called.
    :param params: The parameters to be passed to the function.
    :param user_api_keys: A list of the user's API keys.
    :return: The response from the API.
    """
    provider_config = get_api_config(provider_name)
    function_config = provider_config["functions"][function_name]

    # Use the user's API key if it is available
    api_key = provider_config["api_key"]
    for user_api_key in user_api_keys:
        if user_api_key["provider"] == provider_name:
            api_key = user_api_key["key"]
            break

    # Add the API key to the parameters
    params[provider_config["api_key_param"]] = api_key

    # Add the function name to the parameters
    if "function_param" in function_config:
        params[function_config["function_param"]] = function_config["function_name"]

    response = requests.get(provider_config["base_url"], params=params)
    response.raise_for_status()

    # Handle different response types
    if function_config.get("response_type") == "time_series":
        return response.json()[function_config["data_key"]]
    else:
        return response.json()

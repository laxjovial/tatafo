# domain_tools/weather_tools/__init__.py

from .weather_tool import (
    get_current_weather,
    get_weather_forecast,
    get_air_quality,
    weather_search_web,
    weather_query_uploaded_docs,
    weather_summarize_document_by_path
)

# You can optionally create a class to group these tools if needed
class WeatherTools:
    def __init__(self):
        self.get_current_weather = get_current_weather
        self.get_weather_forecast = get_weather_forecast
        self.get_air_quality = get_air_quality
        self.weather_search_web = weather_search_web
        self.weather_query_uploaded_docs = weather_query_uploaded_docs
        self.weather_summarize_document_by_path = weather_summarize_document_by_path

    def get_tools(self):
        return [
            self.get_current_weather,
            self.get_weather_forecast,
            self.get_air_quality,
            self.weather_search_web,
            self.weather_query_uploaded_docs,
            self.weather_summarize_document_by_path
        ]


# domain_tools/travel_tools/__init__.py

from .travel_tool import (
    search_flights,
    search_hotels,
    get_destination_info,
    travel_search_web,
    travel_query_uploaded_docs,
    travel_summarize_document_by_path
)

# You can optionally create a class to group these tools if needed
class TravelTools:
    def __init__(self):
        self.search_flights = search_flights
        self.search_hotels = search_hotels
        self.get_destination_info = get_destination_info
        self.travel_search_web = travel_search_web
        self.travel_query_uploaded_docs = travel_query_uploaded_docs
        self.travel_summarize_document_by_path = travel_summarize_document_by_path

    def get_tools(self):
        return [
            self.search_flights,
            self.search_hotels,
            self.get_destination_info,
            self.travel_search_web,
            self.travel_query_uploaded_docs,
            self.travel_summarize_document_by_path
        ]



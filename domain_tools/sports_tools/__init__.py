# domain_tools/sports_tools/__init__.py

from .sports_tool import (
    get_latest_scores,
    get_upcoming_events,
    sports_search_web,
    sports_query_uploaded_docs,
    sports_summarize_document_by_path
)

# You can optionally create a class to group these tools if needed
class SportsTools:
    def __init__(self):
        self.get_latest_scores = get_latest_scores
        self.get_upcoming_events = get_upcoming_events
        self.sports_search_web = sports_search_web
        self.sports_query_uploaded_docs = sports_query_uploaded_docs
        self.sports_summarize_document_by_path = sports_summarize_document_by_path

    def get_tools(self):
        return [
            self.get_latest_scores,
            self.get_upcoming_events,
            self.sports_search_web,
            self.sports_query_uploaded_docs,
            self.sports_summarize_document_by_path
        ]



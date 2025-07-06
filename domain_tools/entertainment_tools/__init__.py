# domain_tools/entertainment_tools/__init__.py

from .entertainment_tool import (
    search_movies,
    search_tv_shows,
    entertainment_search_web,
    entertainment_query_uploaded_docs,
    entertainment_summarize_document_by_path
)

# You can optionally create a class to group these tools if needed
class EntertainmentTools:
    def __init__(self):
        self.search_movies = search_movies
        self.search_tv_shows = search_tv_shows
        self.entertainment_search_web = entertainment_search_web
        self.entertainment_query_uploaded_docs = entertainment_query_uploaded_docs
        self.entertainment_summarize_document_by_path = entertainment_summarize_document_by_path

    def get_tools(self):
        return [
            self.search_movies,
            self.search_tv_shows,
            self.entertainment_search_web,
            self.entertainment_query_uploaded_docs,
            self.entertainment_summarize_document_by_path
        ]




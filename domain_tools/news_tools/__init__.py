# domain_tools/news_tools/__init__.py

from .news_tool import (
    get_top_headlines,
    search_news, # Corrected from search_news_articles
    news_search_web,
    news_query_uploaded_docs,
    news_summarize_document_by_path
)

# You can optionally create a class to group these tools if needed
class NewsTools:
    def __init__(self):
        self.get_top_headlines = get_top_headlines
        self.search_news = search_news # Corrected from search_news_articles
        self.news_search_web = news_search_web
        self.news_query_uploaded_docs = news_query_uploaded_docs
        self.news_summarize_document_by_path = news_summarize_document_by_path

    def get_tools(self):
        return [
            self.get_top_headlines,
            self.search_news,
            self.news_search_web,
            self.news_query_uploaded_docs,
            self.news_summarize_document_by_path
        ]


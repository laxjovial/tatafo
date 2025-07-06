# domain_tools/legal_tools/__init__.py

from .legal_tool import (
    perform_legal_research, # Corrected from get_legal_definition
    legal_search_web,
    legal_query_uploaded_docs,
    legal_summarize_document_by_path
)

# You can optionally create a class to group these tools if needed
class LegalTools:
    def __init__(self):
        self.perform_legal_research = perform_legal_research
        self.legal_search_web = legal_search_web
        self.legal_query_uploaded_docs = legal_query_uploaded_docs
        self.legal_summarize_document_by_path = legal_summarize_document_by_path

    def get_tools(self):
        return [
            self.perform_legal_research,
            self.legal_search_web,
            self.legal_query_uploaded_docs,
            self.legal_summarize_document_by_path
        ]


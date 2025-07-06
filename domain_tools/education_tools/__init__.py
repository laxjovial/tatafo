# domain_tools/education_tools/__init__.py

from .education_tool import (
    search_educational_resources, # Corrected from search_educational_courses
    education_search_web,
    education_query_uploaded_docs,
    education_summarize_document_by_path
)

# You can optionally create a class to group these tools if needed
class EducationTools:
    def __init__(self):
        self.search_educational_resources = search_educational_resources # Corrected from search_educational_courses
        self.education_search_web = education_search_web
        self.education_query_uploaded_docs = education_query_uploaded_docs
        self.education_summarize_document_by_path = education_summarize_document_by_path

    def get_tools(self):
        return [
            self.search_educational_resources, # Corrected from search_educational_courses
            self.education_search_web,
            self.education_query_uploaded_docs,
            self.education_summarize_document_by_path
        ]


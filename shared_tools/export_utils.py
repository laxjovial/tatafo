# shared_tools/export_utils.py

import logging
from pathlib import Path
import json
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Base directory for all exports
BASE_EXPORT_DIR = Path("exports")

def export_response(
    content: str,
    user_token: str,
    file_prefix: str = "response",
    file_extension: str = "md"
) -> str:
    """
    Exports a string content (e.g., chat response, summary) to a file
    within a user-specific export directory.

    Args:
        content (str): The string content to export.
        user_token (str): The unique identifier for the user, used for the export path.
        file_prefix (str, optional): A prefix for the filename. Defaults to "response".
        file_extension (str, optional): The file extension. Defaults to "md" (Markdown).

    Returns:
        str: The full path to the exported file, or an error message.
    """
    logger.info(f"Exporting content for user: {user_token} with prefix: {file_prefix}")

    user_export_dir = BASE_EXPORT_DIR / user_token
    user_export_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists

    # Generate a unique filename to prevent overwrites
    filename = f"{file_prefix}_{Path(str(user_export_dir)).stem}_{Path(str(user_export_dir)).name}_{len(list(user_export_dir.iterdir())) + 1}.{file_extension}"
    file_path = user_export_dir / filename

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Content exported successfully to: {file_path}")
        return f"Content exported to: `{file_path}`"
    except Exception as e:
        logger.error(f"Error exporting content to {file_path}: {e}", exc_info=True)
        return f"Error exporting content: {e}"

def export_vector_results(
    results: List[Dict[str, Any]],
    user_token: str,
    section: str,
    file_prefix: str = "vector_results"
) -> str:
    """
    Exports a list of vector search results (e.g., document chunks) to a Markdown file.

    Args:
        results (List[Dict[str, Any]]): A list of dictionaries, where each dictionary
                                         represents a retrieved document chunk (e.g., from RAG).
                                         Expected keys: 'page_content', 'metadata'.
        user_token (str): The unique identifier for the user.
        section (str): The application section (e.g., "medical", "legal").
        file_prefix (str, optional): A prefix for the filename. Defaults to "vector_results".

    Returns:
        str: The full path to the exported file, or an error message.
    """
    logger.info(f"Exporting vector results for user: {user_token}, section: {section}")

    user_export_dir = BASE_EXPORT_DIR / user_token / section
    user_export_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists

    filename = f"{file_prefix}_{Path(str(user_export_dir)).stem}_{Path(str(user_export_dir)).name}_{len(list(user_export_dir.iterdir())) + 1}.md"
    file_path = user_export_dir / filename

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"# Vector Search Results for {section.capitalize()} (User: {user_token})\n\n")
            f.write(f"Query Time: {firestore.SERVER_TIMESTAMP}\n\n") # Placeholder for actual query time
            for i, result in enumerate(results):
                f.write(f"## Result {i+1}\n")
                f.write(f"**Source:** {result.get('metadata', {}).get('source', 'N/A')}\n")
                f.write(f"**Chunk Index:** {result.get('metadata', {}).get('chunk_idx', 'N/A')}\n")
                f.write(f"**Content:**\n```\n{result.get('page_content', 'N/A')}\n```\n\n")
                f.write("---\n\n")
        logger.info(f"Vector results exported successfully to: {file_path}")
        return f"Vector search results exported to: `{file_path}`"
    except Exception as e:
        logger.error(f"Error exporting vector results to {file_path}: {e}", exc_info=True)
        return f"Error exporting vector results: {e}"

# CLI Test (optional)
if __name__ == "__main__":
    import shutil
    import os
    from firebase_admin import firestore # Required for firestore.SERVER_TIMESTAMP in mock

    # Mock firestore.SERVER_TIMESTAMP for local testing
    class MockFirestore:
        SERVER_TIMESTAMP = "MOCK_TIMESTAMP"
    firestore = MockFirestore()

    logging.basicConfig(level=logging.INFO)

    test_user = "test_user_export"
    test_section = "test_section_export"

    # Clean up exports directory from previous runs
    if BASE_EXPORT_DIR.exists():
        shutil.rmtree(BASE_EXPORT_DIR)
    BASE_EXPORT_DIR.mkdir(exist_ok=True)

    print("\n--- Testing export_response function ---")
    content_to_export = "This is a sample chat response that needs to be exported."
    export_path = export_response(content_to_export, test_user, "chat_log", "txt")
    print(f"Export response result: {export_path}")
    expected_path_part = f"exports/{test_user}/chat_log_{test_user}_{test_user}_1.txt"
    assert expected_path_part in export_path
    assert Path(export_path.replace("Content exported to: `", "").replace("`", "")).exists()
    print("Test 1 Passed: export_response created file.")

    # Test with different extension
    content_to_export_md = "# My Report\n\nThis is a markdown report."
    export_path_md = export_response(content_to_export_md, test_user, "report", "md")
    print(f"Export markdown result: {export_path_md}")
    expected_path_part_md = f"exports/{test_user}/report_{test_user}_{test_user}_2.md"
    assert expected_path_part_md in export_path_md
    assert Path(export_path_md.replace("Content exported to: `", "").replace("`", "")).exists()
    print("Test 2 Passed: export_response created markdown file.")


    print("\n--- Testing export_vector_results function ---")
    sample_vector_results = [
        {"page_content": "This is the first chunk of a document. It contains important information about topic A.", "metadata": {"source": "doc1.pdf", "chunk_idx": 0}},
        {"page_content": "The second chunk continues the discussion on topic A and introduces topic B.", "metadata": {"source": "doc1.pdf", "chunk_idx": 1}},
        {"page_content": "A third chunk from a different document, related to topic B.", "metadata": {"source": "doc2.docx", "chunk_idx": 0}},
    ]
    vector_export_path = export_vector_results(sample_vector_results, test_user, test_section, "medical_search")
    print(f"Export vector results result: {vector_export_path}")
    expected_vector_path_part = f"exports/{test_user}/{test_section}/medical_search_{test_user}_{test_section}_1.md"
    assert expected_vector_path_part in vector_export_path
    assert Path(vector_export_path.replace("Vector search results exported to: `", "").replace("`", "")).exists()
    print("Test 3 Passed: export_vector_results created file.")

    # Test with empty results
    print("\n--- Test 4: Export empty vector results ---")
    empty_vector_export_path = export_vector_results([], test_user, test_section, "empty_results")
    print(f"Export empty vector results result: {empty_vector_export_path}")
    assert "Error exporting vector results" not in empty_vector_export_path # Should still create file, just empty content
    empty_file_path = Path(empty_vector_export_path.replace("Vector search results exported to: `", "").replace("`", ""))
    assert empty_file_path.exists()
    with open(empty_file_path, 'r') as f:
        content = f.read()
        assert "Result 1" not in content # Should not have results
    print("Test 4 Passed: Exported empty vector results gracefully.")


    print("\nAll export_utils tests passed.")

    # Clean up exports directory
    if BASE_EXPORT_DIR.exists():
        shutil.rmtree(BASE_EXPORT_DIR)
        print(f"\nCleaned up exports directory: {BASE_EXPORT_DIR}")

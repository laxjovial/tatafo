# shared_tools/export_utils.py

import logging
from pathlib import Path
import json
from typing import List, Dict, Any, Optional
import pandas as pd # For exporting DataFrames
import shutil # For copying files

# Import UserProfile for type hinting
from backend.models.user_models import UserProfile

logger = logging.getLogger(__name__)

# Base directory for all exports
BASE_EXPORT_DIR = Path("exports")

def _generate_unique_filepath(user_id: str, section: Optional[str] = None, file_prefix: str = "export", file_extension: str = "txt") -> Path:
    """
    Generates a unique file path within the user's specific export directory.
    """
    if section:
        user_export_dir = BASE_EXPORT_DIR / user_id / section
    else:
        user_export_dir = BASE_EXPORT_DIR / user_id
    
    user_export_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists

    # Generate a unique filename to prevent overwrites
    # Using a combination of prefix, user_id, and a counter based on existing files in the directory
    # A more robust solution might use a timestamp or UUID, but this is simpler for demonstration.
    # Using UUID for better uniqueness
    filename = f"{file_prefix}_{uuid.uuid4().hex}.{file_extension}"
    file_path = user_export_dir / filename
    return file_path


def export_response(
    content: str,
    user_context: UserProfile, # Changed to UserProfile
    file_prefix: str = "response",
    file_extension: str = "md"
) -> str:
    """
    Exports a string content (e.g., chat response, summary) to a file
    within a user-specific export directory.

    Args:
        content (str): The string content to export.
        user_context (UserProfile): The user's profile, used for the export path.
        file_prefix (str, optional): A prefix for the filename. Defaults to "response".
        file_extension (str, optional): The file extension. Defaults to "md" (Markdown).

    Returns:
        str: The full path to the exported file, or an error message.
    """
    user_id = user_context.user_id
    logger.info(f"Exporting content for user: {user_id} with prefix: {file_prefix}")

    file_path = _generate_unique_filepath(user_id, file_prefix=file_prefix, file_extension=file_extension)

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
    user_context: UserProfile, # Changed to UserProfile
    section: str,
    file_prefix: str = "vector_results"
) -> str:
    """
    Exports a list of vector search results (e.g., document chunks) to a Markdown file.

    Args:
        results (List[Dict[str, Any]]): A list of dictionaries, where each dictionary
                                         represents a retrieved document chunk (e.g., from RAG).
                                         Expected keys: 'page_content', 'metadata'.
        user_context (UserProfile): The user's profile.
        section (str): The application section (e.g., "medical", "legal").
        file_prefix (str, optional): A prefix for the filename. Defaults to "vector_results".

    Returns:
        str: The full path to the exported file, or an error message.
    """
    user_id = user_context.user_id
    logger.info(f"Exporting vector results for user: {user_id}, section: {section}")

    file_path = _generate_unique_filepath(user_id, section=section, file_prefix=file_prefix, file_extension="md")

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"# Vector Search Results for {section.capitalize()} (User: {user_id})\n\n")
            # Mock firestore.SERVER_TIMESTAMP for local testing if not available
            try:
                from firebase_admin import firestore # Try to import if available
                timestamp = firestore.SERVER_TIMESTAMP
            except (ImportError, AttributeError):
                timestamp = datetime.now(timezone.utc).isoformat() # Fallback for local tests
            f.write(f"Query Time: {timestamp}\n\n")
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

def export_chart_file(
    source_chart_path: str,
    user_context: UserProfile, # Changed to UserProfile
    chart_type: str = "chart",
    export_format: str = "png"
) -> str:
    """
    Exports a generated chart file (image or HTML) to the user's specific export directory.

    Args:
        source_chart_path (str): The temporary path where the chart file was initially saved.
        user_context (UserProfile): The user's profile.
        chart_type (str, optional): A descriptive type for the chart (e.g., "line_chart", "histogram"). Defaults to "chart".
        export_format (str, optional): The format of the chart file (e.g., "png", "html"). Defaults to "png".

    Returns:
        str: The full path to the exported chart file, or an error message.
    """
    user_id = user_context.user_id
    logger.info(f"Exporting chart file for user: {user_id}, type: {chart_type}")

    source_path = Path(source_chart_path)
    if not source_path.exists():
        return f"Error: Source chart file not found at '{source_chart_path}'."

    # Use a specific subdirectory for charts within the user's export folder
    chart_export_dir = BASE_EXPORT_DIR / user_id / "charts"
    chart_export_dir.mkdir(parents=True, exist_ok=True)

    # Generate a unique filename for the exported chart
    filename = f"{chart_type}_{uuid.uuid4().hex}.{export_format}"
    destination_path = chart_export_dir / filename

    try:
        shutil.copy(source_path, destination_path)
        logger.info(f"Chart file copied successfully from {source_chart_path} to: {destination_path}")
        # Optionally, remove the temporary source file if it's no longer needed
        # os.remove(source_path)
        return f"Chart exported to: `{destination_path}`"
    except Exception as e:
        logger.error(f"Error exporting chart file from {source_chart_path} to {destination_path}: {e}", exc_info=True)
        return f"Error exporting chart file: {e}"

def export_dataframe_to_file(
    data_json: str,
    user_context: UserProfile,
    file_prefix: str = "data_export",
    export_format: str = "csv" # "csv", "json"
) -> str:
    """
    Exports a JSON string of data (representing a DataFrame) to a file (CSV or JSON).

    Args:
        data_json (str): A JSON string representing the data (list of dictionaries).
        user_context (UserProfile): The user's profile.
        file_prefix (str, optional): A prefix for the filename. Defaults to "data_export".
        export_format (str, optional): The desired export format ("csv" or "json"). Defaults to "csv".

    Returns:
        str: The full path to the exported file, or an error message.
    """
    user_id = user_context.user_id
    logger.info(f"Exporting data for user: {user_id} to {export_format}")

    file_path = _generate_unique_filepath(user_id, file_prefix=file_prefix, file_extension=export_format)

    try:
        data = json.loads(data_json)
        if not isinstance(data, list) or not all(isinstance(d, dict) for d in data):
            raise ValueError("Input data_json must be a JSON array of objects.")
        
        if not data:
            return f"Error: No data provided to export to {export_format}."

        df = pd.DataFrame(data)

        if export_format == "csv":
            df.to_csv(file_path, index=False, encoding="utf-8")
        elif export_format == "json":
            df.to_json(file_path, orient="records", indent=4, encoding="utf-8")
        else:
            return f"Error: Unsupported export format '{export_format}'. Supported: 'csv', 'json'."

        logger.info(f"Data exported successfully to: {file_path}")
        return f"Data exported to: `{file_path}`"
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON data provided for export: {data_json}", exc_info=True)
        return "Error: Invalid JSON data provided for data export."
    except ValueError as ve:
        logger.error(f"Data processing error for export: {ve}", exc_info=True)
        return f"Error: Data processing failed for data export: {ve}"
    except Exception as e:
        logger.error(f"An unexpected error occurred during data export: {e}", exc_info=True)
        return f"An unexpected error occurred during data export: {e}"


# CLI Test (optional)
if __name__ == "__main__":
    import shutil
    import os
    from datetime import datetime, timezone
    from unittest.mock import MagicMock, patch

    # Mock firestore.SERVER_TIMESTAMP for local testing
    class MockFirestore:
        SERVER_TIMESTAMP = datetime.now(timezone.utc).isoformat()
    # Patch the firestore import within this module's scope for testing
    with patch('shared_tools.export_utils.firestore', new=MockFirestore()):
        logging.basicConfig(level=logging.INFO)

        # Mock UserProfile for testing
        mock_user_profile = UserProfile(user_id="test_user_export", username="TestUser", email="test@example.com", tier="pro", roles=["user"])

        # Clean up exports directory from previous runs
        if BASE_EXPORT_DIR.exists():
            shutil.rmtree(BASE_EXPORT_DIR)
        BASE_EXPORT_DIR.mkdir(exist_ok=True)

        print("\n--- Testing export_response function ---")
        content_to_export = "This is a sample chat response that needs to be exported."
        export_path = export_response(content_to_export, mock_user_profile, "chat_log", "txt")
        print(f"Export response result: {export_path}")
        expected_path_part = f"exports/{mock_user_profile.user_id}/chat_log_{uuid.UUID(export_path.split('_')[-1].split('.')[0]).hex}.txt" # Use UUID part for assertion
        assert isinstance(export_path, str) and Path(export_path.replace("Content exported to: `", "").replace("`", "")).exists()
        assert mock_user_profile.user_id in export_path
        print("Test 1 Passed: export_response created file.")

        print("\n--- Testing export_vector_results function ---")
        sample_vector_results = [
            {"page_content": "This is the first chunk of a document.", "metadata": {"source": "doc1.pdf", "chunk_idx": 0}},
            {"page_content": "The second chunk continues the discussion.", "metadata": {"source": "doc1.pdf", "chunk_idx": 1}},
        ]
        vector_export_path = export_vector_results(sample_vector_results, mock_user_profile, "medical", "medical_search")
        print(f"Export vector results result: {vector_export_path}")
        assert isinstance(vector_export_path, str) and Path(vector_export_path.replace("Vector search results exported to: `", "").replace("`", "")).exists()
        assert mock_user_profile.user_id in vector_export_path
        assert "medical" in vector_export_path
        print("Test 2 Passed: export_vector_results created file.")

        print("\n--- Testing export_chart_file function ---")
        # Create a dummy chart file to be exported
        dummy_chart_dir = Path("temp_charts")
        dummy_chart_dir.mkdir(exist_ok=True)
        dummy_chart_path = dummy_chart_dir / "temp_chart.png"
        with open(dummy_chart_path, "w") as f:
            f.write("dummy chart content") # Simulate a chart file

        chart_export_path = export_chart_file(str(dummy_chart_path), mock_user_profile, "line_chart", "png")
        print(f"Export chart result: {chart_export_path}")
        assert isinstance(chart_export_path, str) and Path(chart_export_path.replace("Chart exported to: `", "").replace("`", "")).exists()
        assert mock_user_profile.user_id in chart_export_path
        assert "charts" in chart_export_path
        assert "line_chart" in chart_export_path
        print("Test 3 Passed: export_chart_file copied chart.")

        # Test HTML chart export
        dummy_html_chart_path = dummy_chart_dir / "temp_chart.html"
        with open(dummy_html_chart_path, "w") as f:
            f.write("<html><body><p>dummy html chart</p></body></html>")

        html_chart_export_path = export_chart_file(str(dummy_html_chart_path), mock_user_profile, "plotly_chart", "html")
        print(f"Export HTML chart result: {html_chart_export_path}")
        assert isinstance(html_chart_export_path, str) and Path(html_chart_export_path.replace("Chart exported to: `", "").replace("`", "")).exists()
        assert html_chart_export_path.endswith(".html")
        print("Test 4 Passed: export_chart_file copied HTML chart.")

        print("\n--- Testing export_dataframe_to_file function ---")
        sample_df_data = [
            {"col1": 1, "col2": "A"},
            {"col1": 2, "col2": "B"},
            {"col1": 3, "col2": "C"},
        ]
        sample_df_json = json.dumps(sample_df_data)

        # Export to CSV
        csv_export_path = export_dataframe_to_file(sample_df_json, mock_user_profile, "my_data", "csv")
        print(f"Export CSV result: {csv_export_path}")
        assert isinstance(csv_export_path, str) and Path(csv_export_path.replace("Data exported to: `", "").replace("`", "")).exists()
        assert csv_export_path.endswith(".csv")
        with open(Path(csv_export_path.replace("Data exported to: `", "").replace("`", "")), 'r') as f:
            content = f.read()
            assert "col1,col2" in content
            assert "1,A" in content
        print("Test 5 Passed: export_dataframe_to_file exported to CSV.")

        # Export to JSON
        json_export_path = export_dataframe_to_file(sample_df_json, mock_user_profile, "my_data", "json")
        print(f"Export JSON result: {json_export_path}")
        assert isinstance(json_export_path, str) and Path(json_export_path.replace("Data exported to: `", "").replace("`", "")).exists()
        assert json_export_path.endswith(".json")
        with open(Path(json_export_path.replace("Data exported to: `", "").replace("`", "")), 'r') as f:
            content = f.read()
            assert '"col1": 1' in content
            assert '"col2": "A"' in content
        print("Test 6 Passed: export_dataframe_to_file exported to JSON.")

        # Test unsupported format
        unsupported_format_result = export_dataframe_to_file(sample_df_json, mock_user_profile, "my_data", "xlsx")
        print(f"Unsupported format result: {unsupported_format_result}")
        assert "Error: Unsupported export format 'xlsx'" in unsupported_format_result
        print("Test 7 Passed: Unsupported format handled.")


        print("\nAll export_utils tests passed.")

        # Clean up all created directories and files
        if BASE_EXPORT_DIR.exists():
            shutil.rmtree(BASE_EXPORT_DIR)
            print(f"\nCleaned up exports directory: {BASE_EXPORT_DIR}")
        if dummy_chart_dir.exists():
            shutil.rmtree(dummy_chart_dir)
            print(f"Cleaned up dummy chart directory: {dummy_chart_dir}")

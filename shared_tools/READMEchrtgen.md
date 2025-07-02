Implementing shared_tools/ Directory - Part 2
Now, we will implement the chart_generation_tool.py, which is crucial for enabling chart visualization and exportability.

1. shared_tools/chart_generation_tool.py (NEW FILE)
This new file will provide a tool for generating charts from data (e.g., in JSON format) and saving them as image files. The tool will return the path to the generated image, which the frontend can then display and offer for download.

Important Considerations:

Libraries: This tool will typically rely on plotting libraries like matplotlib, seaborn, or plotly. For this implementation, we'll use matplotlib for simplicity, as it's a common choice and doesn't require external services.

Data Format: The tool expects data in a format it can easily plot, such as a list of dictionaries or a dictionary of lists.

Output: It saves the chart as an image file (e.g., PNG) and returns the file path.

RBAC: The use of this tool will be gated by the chart_generation_enabled capability, and the export of charts by chart_export_enabled, as defined in data/rbac_capabilities.yaml.

shared_tools/chart_generation_tool.py
Jul 2, 7:45 AM

Open
Key Features of shared_tools/chart_generation_tool.py:

Chart Generation: Uses matplotlib to create line, bar, and scatter charts from JSON data.

Dynamic File Paths: Generates unique filenames and stores charts in user-specific subdirectories within a charts/ base directory.

RBAC Integration: Includes a direct check using get_user_tier_capability to ensure that chart generation is enabled for the current user's tier/roles. This prevents unauthorized chart creation. The export_enabled flag passed to the tool is intended to reflect the chart_export_enabled capability, allowing the calling UI/agent to decide if a download link should be provided.

Error Handling: Catches JSONDecodeError, ValueError, and general exceptions, returning informative error messages.

Resource Management: plt.close() is used to free up memory after saving the plot.

Test Suite: The if __name__ == "__main__": block includes comprehensive tests for various scenarios, including RBAC checks, invalid inputs, and different chart types. It also includes mocks for st.secrets, config_manager, and user_manager to allow standalone testing.

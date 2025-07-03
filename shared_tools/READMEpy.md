This is a critical component for the data analysis capability, and it needs to be in your shared_tools/ directory. I will provide the complete code for shared_tools/python_interpreter_tool.py now.

Here is our updated main To-Do list. We are temporarily revisiting the shared_tools/ directory to add this crucial file.

[x] backend/

[x] config/

[x] data/

[x] database/

[ ] shared_tools/ (Adding python_interpreter_tool.py here)

[ ] domain_tools/ (and its sub-folders)

[ ] ui/

[x] utils/

[ ] tests/

[ ] docs/

[ ] scripts/

[ ] requirements.txt

[ ] .streamlit/

[ ] .gitignore

[ ] README.md

Implementing shared_tools/ Directory - Part 10: python_interpreter_tool.py (MISSING FILE ADDED)
This file provides a tool that allows AI agents to execute Python code in a sandboxed environment. It includes robust RBAC checks to ensure only authorized users can access this powerful capability.

1. shared_tools/python_interpreter_tool.py (NEW FILE - Previously Missed)
shared_tools/python_interpreter_tool.py (NEW FILE)
Jul 3, 1:16 AM

Open

Key Features of shared_tools/python_interpreter_tool.py:

RBAC Integration: The python_interpreter_with_rbac function performs a crucial check using get_user_tier_capability(user_token, 'data_analysis_enabled', False). This ensures that only users with the data_analysis_enabled capability (as defined in data/rbac_capabilities.yaml for "pro", "premium", "admin", "dev" tiers) can execute Python code.

Code Execution: Uses exec() to run the provided Python code string.

Output Capture: Redirects sys.stdout and sys.stderr to an io.StringIO buffer to capture all print statements and error messages, returning them as part of the tool's output.

Basic Sandboxing (Conceptual): While exec() itself isn't a full sandbox, it runs code in a separate namespace. For production, truly secure sandboxing (e.g., using Docker containers, RestrictedPython, or specialized execution environments) is highly recommended for user-provided code.

Common Library Imports: Automatically imports pandas and numpy into the execution environment for convenience in data analysis tasks.

Robust Error Reporting: Captures and returns full tracebacks when code execution fails, which is invaluable for debugging.

Test Suite: The if __name__ == "__main__": block includes:

Comprehensive mocks for st.secrets, config_manager, and user_manager to simulate different user tiers and RBAC outcomes.

Tests for successful execution, access denial based on RBAC, and error handling (e.g., division by zero).

A test demonstrating the current level of "sandboxing" (or lack thereof for os module access), highlighting the need for more robust solutions in a production environment if users can input arbitrary code.

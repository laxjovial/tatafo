. Which files make use of the python_interpreter_tool.py and what do their imports look like?
The python_interpreter_tool.py is primarily used by the LLM agent itself. It's not directly imported by most UI files. Instead, it's passed as one of the available "tools" to the LLM when the LLM needs to perform a task like data analysis.

The main files that orchestrate this are:

utils/llm_manager.py (for the Streamlit frontend)

backend/services/llm_service.py (for the FastAPI backend)

These files are responsible for:

Initializing the LLM.

Defining the system prompt for the LLM (which tells it about its capabilities and available tools).

Passing a list of callable tools (including python_interpreter_with_rbac) to the LLM's agent executor.

Here's how the import and usage would look in utils/llm_manager.py (the Streamlit-side LLM orchestrator) and backend/services/llm_service.py:

A. In utils/llm_manager.py (Streamlit Frontend):

This file's chat_with_agent method dynamically determines which tools are available to the LLM based on the user's tier.

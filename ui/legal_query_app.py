import streamlit as st
import requests
import json

# --- Configuration ---
# Replace with the actual URL of your backend server
# For local development, it might be something like "http://localhost:8000"
# For deployment, it will be the public URL of your backend service.
BACKEND_URL = "http://localhost:8000" # IMPORTANT: Update this to your actual backend URL

# --- Streamlit Page Setup ---
st.set_page_config(
    page_title="Legal Query App",
    page_icon="⚖️",
    layout="centered"
)

st.title("⚖️ Legal Information Query")
st.markdown(
    """
    This application allows you to query legal information by interacting directly
    with the backend's legal tools.
    """
)

st.divider()

# --- Input Form ---
st.header("Query Parameters")

# Text input for legal topic
legal_topic = st.text_input(
    "Enter Legal Topic (Optional):",
    placeholder="e.g., 'contract law', 'intellectual property', 'divorce'"
)

# Text area for specific question
specific_question = st.text_area(
    "Enter Specific Question (Optional):",
    placeholder="e.g., 'What are the elements of a valid contract?', 'How to copyright a song?'"
)

# Button to submit the query
if st.button("Get Legal Information", type="primary"):
    # --- Input Validation ---
    if not legal_topic and not specific_question:
        st.warning("Please enter either a legal topic or a specific question to get information.")
    else:
        # --- Construct Tool Invocation Payload ---
        # The tool_name will be 'legal_tools.get_legal_info'
        # The tool_args will be a dictionary containing 'topic' and 'question'
        # Only include arguments that have been provided by the user.
        tool_args = {}
        if legal_topic:
            tool_args["topic"] = legal_topic
        if specific_question:
            tool_args["question"] = specific_question

        payload = {
            "tool_name": "legal_tools.get_legal_info",
            "tool_args": tool_args
        }

        st.info("Sending request to backend...")
        st.json(payload) # Display the payload being sent for debugging/transparency

        # --- Make API Call to Backend ---
        try:
            response = requests.post(
                f"{BACKEND_URL}/tools/invoke",
                json=payload,
                timeout=60 # Set a timeout for the request
            )

            # --- Process Response ---
            if response.status_code == 200:
                result = response.json()
                st.success("Information Retrieved Successfully!")
                st.subheader("Legal Information:")

                # Display the result in a user-friendly format
                if result and isinstance(result, dict):
                    if "result" in result:
                        # Assuming the backend returns a dictionary with a 'result' key
                        st.write(result["result"])
                    else:
                        st.json(result) # Fallback to showing raw JSON if 'result' key is missing
                else:
                    st.json(result) # Display raw JSON if not a dictionary or empty

            else:
                st.error(f"Error: Backend returned status code {response.status_code}")
                st.json(response.json()) # Display error details from backend

        except requests.exceptions.ConnectionError:
            st.error(
                f"Connection Error: Could not connect to the backend at {BACKEND_URL}. "
                "Please ensure the backend server is running and the URL is correct."
            )
        except requests.exceptions.Timeout:
            st.error("Request timed out. The backend took too long to respond.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

st.divider()
st.caption("Powered by your custom legal tools backend.")

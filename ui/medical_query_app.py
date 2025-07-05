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
    page_title="Medical Query App",
    page_icon="⚕️",
    layout="centered"
)

st.title("⚕️ Medical Information Query")
st.markdown(
    """
    This application allows you to query medical information by interacting directly
    with the backend's medical tools.
    """
)

st.divider()

# --- Input Form ---
st.header("Query Parameters")

# Text input for disease name
disease_name = st.text_input(
    "Enter Disease Name (Optional):",
    placeholder="e.g., 'Influenza', 'Diabetes', 'Hypertension'"
)

# Text area for symptoms
symptoms = st.text_area(
    "Enter Symptoms (Optional, comma-separated):",
    placeholder="e.g., 'fever, cough, sore throat', 'fatigue, frequent urination'"
)

# Button to submit the query
if st.button("Get Medical Information", type="primary"):
    # --- Input Validation ---
    if not disease_name and not symptoms:
        st.warning("Please enter either a disease name or symptoms to get information.")
    else:
        # --- Construct Tool Invocation Payload ---
        # The tool_name will be 'medical_tools.get_disease_info'
        # The tool_args will be a dictionary containing 'disease_name' and 'symptoms'
        # Only include arguments that have been provided by the user.
        tool_args = {}
        if disease_name:
            tool_args["disease_name"] = disease_name
        if symptoms:
            # Split symptoms by comma and strip whitespace for cleaner input
            tool_args["symptoms"] = [s.strip() for s in symptoms.split(',') if s.strip()]

        payload = {
            "tool_name": "medical_tools.get_disease_info",
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
                st.subheader("Medical Information:")

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
st.caption("Powered by your custom medical tools backend.")

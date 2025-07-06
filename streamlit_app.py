    # streamlit_app.py (Place this in your project's root directory, e.g., /workspaces/tatafo/)

    import streamlit as st
    import sys
    from pathlib import Path

    # Add project root to sys.path to allow imports from backend/ and ui/
    current_file_path = Path(__file__).resolve()
    project_root = current_file_path.parents[0] # This is already the root
    sys.path.insert(0, str(project_root))

    # Import your individual Streamlit app modules
    from backend import login_app
    from backend import register_app
    # from backend import user_profile_app # Will uncomment/add as we connect them
    # from backend import ai_assistant_app
    # from backend import admin_dashboard_app

    # Set up Streamlit page configuration
    st.set_page_config(
        page_title="Intelli-Agent",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    # Initialize session state for page navigation if not already present
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Login" # Default page

    # --- Main Navigation Logic ---
    def main_app():
        # Header/Navigation Bar (Optional, but good for consistency)
        st.sidebar.title("Navigation")
        if st.sidebar.button("Login"):
            st.session_state.current_page = "Login"
            st.rerun()
        if st.sidebar.button("Register"):
            st.session_state.current_page = "Register"
            st.rerun()
        # Add more navigation buttons here as you connect more apps
        # if st.sidebar.button("AI Assistant"):
        #     st.session_state.current_page = "AI Assistant"
        #     st.rerun()
        # if st.sidebar.button("User Profile"):
        #     st.session_state.current_page = "User Profile"
        #     st.rerun()
        # if st.sidebar.button("Admin Dashboard"):
        #     st.session_state.current_page = "Admin Dashboard"
        #     st.rerun()

        # Display the current page based on session state
        if st.session_state.current_page == "Login":
            login_app.app()
        elif st.session_state.current_page == "Register":
            register_app.app()
        # elif st.session_state.current_page == "AI Assistant":
        #     ai_assistant_app.app()
        # elif st.session_state.current_page == "User Profile":
        #     user_profile_app.app()
        # elif st.session_state.current_page == "Admin Dashboard":
        #     admin_dashboard_app.app()
        else:
            st.markdown(f"## Welcome to {st.session_state.current_page}!")
            st.write("Please use the sidebar to navigate.")

    if __name__ == "__main__":
        main_app()
    

    

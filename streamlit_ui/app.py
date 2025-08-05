import streamlit as st
import requests

# --- Configuration ---
FASTAPI_BASE_URL = "http://localhost:8000"

st.set_page_config(page_title="Intelli-Agent", layout="wide")

st.title("Intelli-Agent")

def login_page():
    st.header("Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if email and password:
            try:
                response = requests.post(f"{FASTAPI_BASE_URL}/login", json={"email": email, "password": password})
                if response.status_code == 200:
                    st.session_state['user_token'] = response.json()['custom_token']
                    st.session_state['user_uid'] = response.json()['uid']
                    st.success("Logged in successfully!")
                    st.experimental_rerun()
                else:
                    st.error(f"Failed to log in: {response.json().get('detail')}")
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter both email and password.")

if 'user_token' not in st.session_state:
    login_page()
else:
    st.sidebar.title("Navigation")

    # Get user roles to conditionally display the Admin page
    try:
        response = requests.get(
            f"{FASTAPI_BASE_URL}/profile/{st.session_state['user_uid']}",
            headers={"Authorization": f"Bearer {st.session_state['user_token']}"}
        )
        if response.status_code == 200:
            roles = response.json().get('roles', [])
            if 'admin' in roles:
                page = st.sidebar.radio("Go to", ["AI Assistant", "User Profile", "Integrations", "Admin"])
            else:
                page = st.sidebar.radio("Go to", ["AI Assistant", "User Profile", "Integrations"])
        else:
            page = st.sidebar.radio("Go to", ["AI Assistant", "User Profile", "Integrations"])
    except Exception:
        page = st.sidebar.radio("Go to", ["AI Assistant", "User Profile", "Integrations"])


    if st.sidebar.button("Logout"):
        del st.session_state['user_token']
        del st.session_state['user_uid']
        st.experimental_rerun()

    if page == "AI Assistant":
        st.header("AI Assistant")

        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []

        for chat in st.session_state['chat_history']:
            with st.chat_message(chat['role']):
                st.markdown(chat['content'])

        prompt = st.chat_input("Ask the AI assistant...")
        if prompt:
            st.session_state['chat_history'].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.spinner("Thinking..."):
                try:
                    response = requests.post(
                        f"{FASTAPI_BASE_URL}/tools/chat/agent",
                        json={
                            "prompt": prompt,
                            "chat_history": st.session_state['chat_history'],
                            "user_token": st.session_state['user_token']
                        },
                        headers={"Authorization": f"Bearer {st.session_state['user_token']}"}
                    )
                    if response.status_code == 200:
                        ai_response = response.json()['response']
                        st.session_state['chat_history'].append({"role": "assistant", "content": ai_response})
                        with st.chat_message("assistant"):
                            st.markdown(ai_response)
                    else:
                        st.error(f"Failed to get response: {response.json().get('detail')}")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

    elif page == "User Profile":
        st.header("User Profile")

        try:
            response = requests.get(
                f"{FASTAPI_BASE_URL}/profile/{st.session_state['user_uid']}",
                headers={"Authorization": f"Bearer {st.session_state['user_token']}"}
            )
            if response.status_code == 200:
                profile = response.json()
                st.write(f"**Username:** {profile['username']}")
                st.write(f"**Email:** {profile['email']}")
                st.write(f"**Tier:** {profile['tier']}")
                st.write(f"**Roles:** {', '.join(profile['roles'])}")

                st.subheader("Storage Usage")
                storage_usage_percent = (profile['storage_usage_mb'] / profile['storage_limit_mb']) * 100
                st.progress(storage_usage_percent)
                st.write(f"{profile['storage_usage_mb']:.2f}MB / {profile['storage_limit_mb']}MB used")
            else:
                st.error(f"Failed to fetch profile: {response.json().get('detail')}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

    elif page == "Integrations":
        st.header("Integrations")

        st.subheader("Connect to External Services")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Connect to Google Drive"):
                try:
                    response = requests.post(
                        f"{FASTAPI_BASE_URL}/integrations/google-drive/connect/start",
                        headers={"Authorization": f"Bearer {st.session_state['user_token']}"}
                    )
                    if response.status_code == 200:
                        st.success("Redirecting to Google Drive for authentication...")
                        st.write(f"Please visit: {response.json()['authorization_url']}")
                    else:
                        st.error(f"Failed to connect: {response.json().get('detail')}")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

        with col2:
            if st.button("Connect to OneDrive"):
                try:
                    response = requests.post(
                        f"{FASTAPI_BASE_URL}/integrations/one-drive/connect/start",
                        headers={"Authorization": f"Bearer {st.session_state['user_token']}"}
                    )
                    if response.status_code == 200:
                        st.success("Redirecting to OneDrive for authentication...")
                        st.write(f"Please visit: {response.json()['authorization_url']}")
                    else:
                        st.error(f"Failed to connect: {response.json().get('detail')}")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

    elif page == "Admin":
        st.header("Admin Dashboard")
        # Admin Dashboard UI will be implemented here

import streamlit as st
import requests
import json

# --- Page Configuration ---
st.set_page_config(
    page_title="The Project Gutenberg eBook of The Adventures of Sherlock Holmes",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# --- Custom CSS for Minimalist Design ---
st.markdown(
    """
<style>
    /* General styles */
    .stApp {
        background-color: #f0f2f6; /* Light grey background */
    }

    /* Chat bubble styles */
    .stChatMessage {
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid transparent;
        max-width: 85%;
    }

    /* User message style */
    [data-testid="chat-message-container"]:has([data-testid="chat-avatar-user"]) {
       display: flex;
       flex-direction: row-reverse;
       text-align: right;
    }
    
    [data-testid="chat-message-container"]:has([data-testid="chat-avatar-user"]) .stChatMessage {
        background-color: #007bff; /* Blue for user messages */
        color: white;
        border-radius: 10px 10px 0 10px;
    }

    /* Assistant message style */
    [data-testid="chat-message-container"]:has([data-testid="chat-avatar-assistant"]) .stChatMessage {
        background-color: #ffffff; /* White for assistant messages */
        color: #333;
        border: 1px solid #e0e0e0;
        border-radius: 10px 10px 10px 0;
        align-self: flex-start;
    }

    /* Input box style */
    .stTextInput>div>div>input {
        border-radius: 8px;
        border: 1px solid #ccc;
        background-color: #fff;
    }
    .stTextInput>div>div>input:focus {
        border-color: #007bff;
        box-shadow: 0 0 0 2px rgba(0,123,255,.25);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }

    [data-testid="stSidebar"] h2 {
        color: #333;
    }
</style>
""",
    unsafe_allow_html=True,
)


# --- API Communication ---
def get_agi_response(prompt, api_url):
    """
    Sends a prompt to the local AGI server and gets a response.

    Args:
        prompt (str): The user's message.
        api_url (str): The URL of the local AGI server.
        model_name (str): The name of the model to use.

    Returns:
        str: The AGI's response, or an error message.
    """
    try:
        # Most local LLM APIs expect a JSON payload
        payload = {
            "query": prompt,
            "index_name": "new_test",
        }

        # Make the POST request
        response = requests.post(
            api_url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=120,  # Set a timeout for the request
        )
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        # Assuming the API returns a JSON with a 'response' key
        # This might need to be adjusted based on your specific AGI's API format
        # For example, some models return under 'choices'[0]['message']['content']
        response_data = response.json()

        # Common response formats:
        if "answer" in response_data:
            return response_data["answer"]
        elif "choices" in response_data and response_data["choices"]:
            return response_data["choices"][0]["message"]["content"]
        else:
            # Fallback to returning the whole JSON if format is unknown
            return json.dumps(response_data, indent=2)

    except requests.exceptions.RequestException as e:
        return f"Error: Could not connect to the AGI server at {api_url}. Please ensure it's running. Details: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"


# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("🤖 AGI Configuration")
    st.markdown("Connect to your local Artificial General Intelligence model.")

    api_url = st.text_input(
        "AGI API Endpoint",
        value="http://localhost:11434/api/generate",
        help="The URL of your chat bot API endpoint.",
    )


# --- App Initialization ---
st.title("The Project Gutenberg eBook of The Adventures of Sherlock Holmes")
st.markdown(
    "A clean, minimalist chat interface that lets you explore The Adventures of Sherlock Holmes and related literary topics with a local AI model. Ask questions, request summaries, get character analyses, or generate reading suggestions — all while keeping your data on-device."
)

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! How can I help you today?"}
    ]

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input and Response Handling ---
if prompt := st.chat_input("What would you like to ask?"):
    # Add user message to session state and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_agi_response(prompt, api_url)
            st.markdown(response)

    # Add assistant response to session state
    st.session_state.messages.append({"role": "assistant", "content": response})

import streamlit as st
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- Page Configuration ---
st.set_page_config(
    page_title="AI-Powered Translator",
    page_icon="✨",
    layout="centered"
)

# --- Custom CSS for Aesthetics ---
def load_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

        html, body, [class*="st-"] {
            font-family: 'Poppins', sans-serif;
        }

        .stApp {
            background-image: linear-gradient(to right top, #d16ba5, #c777b9, #ba83ca, #aa8fd8, #9a9ae1, #8aa7ec, #79b3f4, #69bff8, #52cffe, #41dfff, #46eefa, #5ffbf1);
            background-attachment: fixed;
            background-size: cover;
        }

        .main-container {
            background-color: rgba(255, 255, 255, 0.9);
            padding: 2rem;
            border-radius: 20px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        }
        
        .stButton button {
            background-color: #69bff8;
            color: white;
            border-radius: 10px;
            border: none;
            padding: 10px 20px;
            font-weight: 600;
            transition: background-color 0.3s;
            margin-top: 1.5rem; /* More space above the button */
        }

        .stButton button:hover {
            background-color: #52cffe;
        }
        
        h1, h3 {
            color: #2c3e50;
        }
        h2 {
            color: #34495e;
            padding-bottom: 0.5rem; /* Space below the subheader */
        }
        </style>
    """, unsafe_allow_html=True)

load_css()

# --- Load Environment Variables and API Key ---
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# --- Header ---
st.title("✨ AI-Powered Translator")
st.markdown("Instantly translate text into any language using the power of LCEL and Groq.")
st.divider() # Add a little vertical space

# --- Check for API Key ---
if not groq_api_key:
    st.error("Groq API Key not found!")
    st.info("""
        **Message for the Developer:**
        To run this application, a Groq API key is required.
        1.  Create a `.env` file in the project's root directory.
        2.  Add your key: `GROQ_API_KEY="gsk_..."`
        3.  Save the file and refresh this page.
    """)
    st.stop()

# --- Core Logic ---
try:
    model = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_api_key)
    prompt = ChatPromptTemplate.from_messages(
        [("system", "Translate the following into {language}:"), ("user", "{text}")]
    )
    parser = StrOutputParser()
    chain = prompt | model | parser
except Exception as e:
    st.error(f"Failed to initialize LangChain components: {e}")
    st.stop()

# --- Main Interaction Container ---
with st.container():
    
    st.subheader("📝 Enter Your Text")

    col1, col2 = st.columns(2)
    with col1:
        target_language = st.text_input("🌐 Target Language", "French")
    with col2:
        text_to_translate = st.text_area("✍️ Text to Translate", "Hello, how are you today?", height=125)

    if st.button("Translate Now!", use_container_width=True):
        if not text_to_translate or not target_language:
            st.warning("Please provide both the target language and the text to translate.")
        else:
            with st.spinner(f"Translating to {target_language}..."):
                try:
                    chain_input = {"language": target_language, "text": text_to_translate}
                    result = chain.invoke(chain_input)
                    
                    st.divider()
                    st.subheader("🎉 Translation Result:")
                    st.markdown(f"### *{result}*")

                except Exception as e:
                    st.error(f"An error occurred during translation: {e}")
                    
    st.markdown('</div>', unsafe_allow_html=True)

# --- Educational Expander ---
st.write("") 
with st.expander(" peek inside the LCEL chain..."):
    st.code("""
# 1. The Prompt Template takes user input
prompt = ChatPromptTemplate.from_messages([
    ("system", "Translate the following into {language}:"), 
    ("user", "{text}")
])

# 2. The Model receives the formatted prompt
model = ChatGroq(model="Llama3-8b-8192", ...)

# 3. The Parser extracts the string content from the model's output
parser = StrOutputParser()

# 4. They are all piped together to form the chain
chain = prompt | model | parser

    """, language="python")

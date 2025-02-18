from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os
from dotenv import load_dotenv
import speech_recognition as sr
from gtts import gTTS
import base64
import tempfile
from streamlit_mic_recorder import mic_recorder  # WORKING MIC INPUT

# Load environment variables
load_dotenv()   

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Initialize Streamlit UI
st.title("🎙️ Voice Chatbot using LLAMA2")

# Initialize LLaMA 2 model
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user queries."),
    ("user", "Question:{question}")
])
llm = Ollama(model="llama2")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# Speech Recognition Function
def recognize_speech(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio_data)  # Convert Speech to Text
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand the audio."
        except sr.RequestError:
            return "Speech recognition service unavailable."

# **Step 1: Live Mic Recording**
st.write("🎤 Press and hold the button below to record your voice:")

audio_data = mic_recorder(start_prompt="🎙️ Click to Speak", stop_prompt="⏹️ Stop Recording", key="mic")

input_text = st.text_input("💬 Or type your question here:")

# **Step 2: Process Recorded Audio**
if audio_data:
    st.audio(audio_data, format="audio/wav")  # Play recorded audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_data)
        temp_audio_path = temp_audio.name

    input_text = recognize_speech(temp_audio_path)  # Convert Speech to Text
    st.write(f"📝 Recognized Text: **{input_text}**")

# **Step 3: Process User Input (Text or Voice)**
if input_text:
    response_text = chain.invoke({"question": input_text})
    st.write("🤖 Response:", response_text)

    # Convert Response to Audio
    tts = gTTS(text=response_text, lang="en")
    temp_audio_path = "response_audio.mp3"
    tts.save(temp_audio_path)

    # Display Audio Player
    with open(temp_audio_path, "rb") as audio_file:
        audio_bytes = audio_file.read()
        audio_base64 = base64.b64encode(audio_bytes).decode()
        audio_html = f"""
        <audio controls>
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
        </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)

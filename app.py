import streamlit as st
import os
import requests
from PIL import Image
import wave
import torch
from transformers import pipeline, BitsAndBytesConfig
from dotenv import load_dotenv
from google import genai
from google.genai import types

# --- Streamlit Page Configuration ---
st.set_page_config(layout="wide", page_title="Radiology with MedGemma & Gemini TTS")

# --- Configuration and API Key Handling ---
# Load environment variables from .env file
load_dotenv()

hf_token = os.getenv("HF_TOKEN")
gemini_api_key = os.getenv("GOOGLE_API_KEY")

if not hf_token:
    st.warning("HF_TOKEN is not set in your .env file or as an environment variable. MedGemma model loading might fail.")
if not gemini_api_key:
    st.warning("GOOGLE_API_KEY is not set in your .env file or as an environment variable. Gemini TTS will not work.")

# Initialize Gemini API Client
gemini_client = None
if gemini_api_key:
    try:
        gemini_client = genai.Client(api_key=gemini_api_key)
    except Exception as e:
        st.error(f"Failed to initialize Gemini API client: {e}")
        st.info("Ensure 'google-generativeai' library is up-to-date and your API key is valid.")
else:
    st.error("Gemini API key is missing. Gemini TTS functionality will be disabled.")


# --- Model Loading (MedGemma) ---
@st.cache_resource
def load_medgemma_model():
    """
    Loads the MedGemma model. Uses st.cache_resource to load only once.
    """
    st.write("Loading MedGemma model... This may take a moment.")
    if not torch.cuda.is_available():
        st.error("CUDA is not available. MedGemma (4b-it) requires a GPU.")
        return None

    try:
        if not hf_token:
            st.error("HF_TOKEN is required to download and load MedGemma. Please set it.")
            return None

        model_kwargs = dict(torch_dtype=torch.bfloat16, device_map="cuda:0", quantization_config=BitsAndBytesConfig(load_in_4bit=True))
        pipe = pipeline("image-text-to-text", model="google/medgemma-4b-it", model_kwargs=model_kwargs)
        pipe.model.generation_config.do_sample = False
        st.success("MedGemma model loaded successfully!")
        return pipe
    except Exception as e:
        st.error(f"Error loading MedGemma model. Ensure HF_TOKEN is correct and you have a compatible GPU: {e}")
        return None

pipe = load_medgemma_model()

# --- Utility Functions ---
def infer(prompt: str, image: Image.Image, system: str = None) -> str:
    """
    Uses MedGemma to generate a plain-language report based on the provided prompt and image.
    """
    if pipe is None:
        st.error("MedGemma model failed to load. Cannot perform inference.")
        return "Error: Model not loaded."

    temp_image_path = "temp_medgemma_input_image.png"
    image.save(temp_image_path)

    messages = []
    if system:
        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": system}]
        })
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image", "image": temp_image_path}
        ]
    })

    try:
        output = pipe(text=messages, max_new_tokens=2048)
        if isinstance(output[0]["generated_text"], list):
            response = output[0]["generated_text"][-1]["content"]
        else:
            response = output[0]["generated_text"]
    except Exception as e:
        st.error(f"Error during MedGemma inference: {e}")
        response = "Error during inference."
    finally:
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
    return response

def wave_file(filename, pcm, channels=1, rate=24000, sample_width=2):
    """
    Converts raw PCM audio data into a proper .wav file.
    """
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm)

# --- Streamlit UI ---
st.title("Demo: Radiology with MedGemma & Gemini's Native TTS")
st.markdown(
    """
        This interactive demo is designed to show you how advanced AI technologies can make complex medical information easier to understand. Here, you’ll experience two of these innovations:
        - **MedGemma**: An advanced AI model built specifically for healthcare, MedGemma can analyze medical images—like X-rays and CT scans—and generate clear, plain-language reports. It’s been trained on a wide range of medical data, allowing it to highlight important findings and explain them in a way that’s accessible to everyone.
        - **Gemini’s Native Text-to-Speech (TTS)**: This technology brings your medical reports to life by reading them out loud in a natural, easy-to-understand voice, making medical insights even more accessible.
        With this demo, you can upload a medical image or provide a link, ask a question or give instructions, and instantly receive both a written and spoken explanation. Our goal is to make medical imaging more approachable and to help users better understand radiology results with the help of cutting-edge AI.
    """
)

with st.sidebar:
    st.header("Input Options")
    source_type = st.radio("Select Image Source:", ["Upload File", "Enter URL"])

    image = None
    if source_type == "Upload File":
        uploaded_file = st.file_uploader("Upload a medical image (e.g., X-ray, CT scan)", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
    else:
        image_url = st.text_input("Enter Image URL:")
        if image_url:
            try:
                response = requests.get(image_url, headers={"User-Agent": "example"}, stream=True)
                response.raise_for_status()
                image = Image.open(response.raw)
                st.image(image, caption="Image from URL", use_container_width=True)
            except requests.exceptions.RequestException as e:
                st.error(f"Error fetching image from URL: {e}. Please check the URL.")
            except Exception as e:
                st.error(f"Error loading image from URL: {e}")

    text_prompt = st.text_area("Instructions (e.g., 'Describe this X-ray in simple terms.')", "Describe this in simple terms.")

st.header("Generated Report")

if st.button("Generate Report and Audio"):
    if image is None:
        st.warning("Please upload an image or provide an image URL.")
    else:
        with st.spinner("Analyzing image and generating report..."):
            try:
                report = infer(text_prompt, image)
                st.subheader("Text Report:")
                st.write(report)

                if gemini_client:
                    st.subheader("Audio Report:")
                    try:
                        tts_response = gemini_client.models.generate_content(
                            model="gemini-2.5-flash-preview-tts",
                            contents=report,
                            config=types.GenerateContentConfig(
                                response_modalities=["AUDIO"],
                                speech_config=types.SpeechConfig(
                                    voice_config=types.VoiceConfig(
                                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                            voice_name='Kore',
                                        )
                                    )
                                ),
                            )
                        )

                        audio_data_raw_pcm = tts_response.candidates[0].content.parts[0].inline_data.data

                        if len(audio_data_raw_pcm) == 0:
                            st.warning("Gemini returned empty audio data. This might happen for very short or invalid prompts, or if the API is experiencing issues.")
                        else:
                            audio_filename = 'generated_report.wav'
                            wave_file(audio_filename, audio_data_raw_pcm) # Pass the raw bytes directly
                            st.audio(audio_filename, format='audio/wav')
                            if os.path.exists(audio_filename):
                                os.remove(audio_filename)
                    except Exception as e:
                        st.error(f"Error generating audio with Gemini TTS: {e}")
                        st.info("Please verify your API key, ensure the 'google-generativeai' library is up-to-date, and check the prompt text.")
                else:
                    st.info("Gemini API client not initialized. Skipping audio generation.")

            except Exception as e:
                st.error(f"An error occurred during report generation: {e}")

st.markdown(
    """
    ---
    ### Disclaimer
   **WARNING: This demonstration does NOT provide medical advice, diagnosis, or treatment. Do NOT rely on any information or results from this tool for your health decisions. Misuse of this demo could lead to serious harm. Always consult a qualified healthcare professional for any medical concerns. If you are experiencing a medical emergency, contact your doctor or call emergency services immediately.**
    """
)

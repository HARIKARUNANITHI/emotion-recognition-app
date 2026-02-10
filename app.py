
import streamlit as st
import google.generativeai as genai
import cv2
import numpy as np
from PIL import Image
import io
import base64
import json
import tempfile
import os
from datetime import datetime
import os

# ============================================
# CONFIGURE YOUR API KEY HERE
# ============================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# ============================================

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Page configuration
st.set_page_config(
    page_title="Emotion Recognition System",
    page_icon="😊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #6B46C1;
        text-align: center;
        margin-bottom: 2rem;
    }
    .emotion-card {
        padding: 2rem;
        border-radius: 1rem;
        border: 2px solid #E2E8F0;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #6B46C1;
        color: white;
        border-radius: 0.5rem;
        padding: 0.75rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #553C9A;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'camera_image' not in st.session_state:
    st.session_state.camera_image = None
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'recorded_audio' not in st.session_state:
    st.session_state.recorded_audio = None
if 'uploaded_audio' not in st.session_state:
    st.session_state.uploaded_audio = None

def analyze_text_emotion(text):
    """Analyze emotion from text"""
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = f"""Analyze the emotional content of the following text. Identify the primary emotion(s) and provide:
1. Primary Emotion (e.g., happy, sad, angry, fearful, surprised, disgusted, neutral)
2. Confidence level (0-100%)
3. Secondary emotions if present (as a comma-separated list)
4. Brief emotional analysis

Text: "{text}"

Respond ONLY with a valid JSON object in this exact format:
{{
    "primaryEmotion": "emotion name",
    "confidence": 85,
    "secondaryEmotions": ["emotion1", "emotion2"],
    "analysis": "brief analysis text"
}}"""
        
        response = model.generate_content(prompt)
        result_text = response.text.strip()
        
        # Extract JSON from response
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()
        
        result = json.loads(result_text)
        result['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result['type'] = 'text'
        return result
        
    except json.JSONDecodeError as e:
        st.error(f"Error parsing response: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error analyzing text: {str(e)}")
        return None

def analyze_image_emotion(image):
    """Analyze emotion from image"""
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = """Analyze the facial expression and emotional state in this image. Provide:
1. Primary Emotion detected from facial expression
2. Confidence level (0-100%)
3. Facial features observed (as a comma-separated list, e.g., smiling, frowning, raised eyebrows)
4. Overall emotional analysis

Respond ONLY with a valid JSON object in this exact format:
{
    "primaryEmotion": "emotion name",
    "confidence": 85,
    "facialFeatures": ["feature1", "feature2", "feature3"],
    "analysis": "brief analysis text"
}"""
        
        response = model.generate_content([prompt, image])
        result_text = response.text.strip()
        
        # Extract JSON from response
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()
        
        result = json.loads(result_text)
        result['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result['type'] = 'image'
        return result
        
    except Exception as e:
        st.error(f"Error analyzing image: {str(e)}")
        return None

def analyze_audio_emotion(audio_file_path):
    """Analyze emotion from audio"""
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Upload audio file
        audio_file = genai.upload_file(path=audio_file_path)
        
        prompt = """Analyze the emotional content and tone from this audio. Provide:
1. Primary Emotion detected
2. Confidence level (0-100%)
3. Vocal characteristics (as a comma-separated list, e.g., tone, pace, intensity)
4. Emotional analysis

This is a speech audio file. Analyze both the content and vocal characteristics.

Respond ONLY with a valid JSON object in this exact format:
{
    "primaryEmotion": "emotion name",
    "confidence": 85,
    "vocalCharacteristics": ["characteristic1", "characteristic2"],
    "analysis": "brief analysis text"
}"""
        
        response = model.generate_content([prompt, audio_file])
        result_text = response.text.strip()
        
        # Extract JSON from response
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()
        
        result = json.loads(result_text)
        result['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result['type'] = 'audio'
        
        # Clean up uploaded file
        genai.delete_file(audio_file.name)
        
        return result
        
    except Exception as e:
        st.error(f"Error analyzing audio: {str(e)}")
        return None

def get_emotion_color(emotion):
    """Get color for emotion display"""
    emotion_lower = emotion.lower()
    colors = {
        'happy': '#FEF3C7',
        'joy': '#FEF3C7',
        'sad': '#DBEAFE',
        'sadness': '#DBEAFE',
        'angry': '#FEE2E2',
        'anger': '#FEE2E2',
        'fearful': '#E9D5FF',
        'fear': '#E9D5FF',
        'surprised': '#FCE7F3',
        'surprise': '#FCE7F3',
        'disgusted': '#D1FAE5',
        'disgust': '#D1FAE5',
        'neutral': '#F3F4F6'
    }
    
    for key, color in colors.items():
        if key in emotion_lower:
            return color
    return '#F3F4F6'

def display_results(result):
    """Display emotion analysis results"""
    if result is None:
        return
    
    st.markdown("---")
    st.markdown("## 📊 Emotion Analysis Results")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Analysis Type:** {result['type'].upper()}")
    with col2:
        st.info(f"**Timestamp:** {result['timestamp']}")
    
    # Primary emotion display
    emotion_color = get_emotion_color(result['primaryEmotion'])
    st.markdown(f"""
    <div style="background-color: {emotion_color}; padding: 2rem; border-radius: 1rem; 
                text-align: center; margin: 1rem 0; border: 3px solid {emotion_color};">
        <h3 style="margin: 0; color: #1F2937;">Primary Emotion</h3>
        <h1 style="margin: 0.5rem 0; color: #1F2937; text-transform: capitalize;">
            {result['primaryEmotion']}
        </h1>
        <p style="margin: 0; color: #4B5563; font-size: 1.1rem;">
            Confidence: {result.get('confidence', 'N/A')}%
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Secondary emotions or features
    if 'secondaryEmotions' in result and result['secondaryEmotions']:
        st.markdown("### 🎭 Secondary Emotions")
        cols = st.columns(len(result['secondaryEmotions']))
        for idx, emotion in enumerate(result['secondaryEmotions']):
            with cols[idx]:
                st.success(emotion.capitalize())
    
    if 'facialFeatures' in result and result['facialFeatures']:
        st.markdown("### 😊 Facial Features Detected")
        cols = st.columns(min(len(result['facialFeatures']), 4))
        for idx, feature in enumerate(result['facialFeatures']):
            with cols[idx % 4]:
                st.info(feature)
    
    if 'vocalCharacteristics' in result and result['vocalCharacteristics']:
        st.markdown("### 🎤 Vocal Characteristics")
        cols = st.columns(min(len(result['vocalCharacteristics']), 3))
        for idx, char in enumerate(result['vocalCharacteristics']):
            with cols[idx % 3]:
                st.warning(char)
    
    # Detailed analysis
    if 'analysis' in result:
        st.markdown("### 📝 Detailed Analysis")
        st.write(result['analysis'])

st.info("Live microphone recording is not supported in online deployment. Please upload an audio file.")


# Main App
def main():
    st.markdown('<div class="main-header">🎭 Real-Time Multimodal Emotion Recognition</div>', 
                unsafe_allow_html=True)
    
    # Main Interface
    st.markdown("### 🎯 Choose Analysis Mode")
    
    tab1, tab2, tab3 = st.tabs(["📝 Text Analysis", "📷 Facial Analysis", "🎤 Voice Analysis"])
    
    # Text Analysis Tab
    with tab1:
        st.markdown("#### Analyze Emotions from Text")
        text_input = st.text_area("Enter text to analyze:", height=150,
                                  placeholder="Type something to analyze its emotional content...")
        
        if st.button("🔍 Analyze Text Emotion", key="text_analyze"):
            if text_input.strip():
                with st.spinner("Analyzing emotions..."):
                    result = analyze_text_emotion(text_input)
                    if result:
                        display_results(result)
            else:
                st.warning("Please enter some text to analyze")
    
    # Facial Analysis Tab
    with tab2:
        st.markdown("#### Analyze Emotions from Facial Expression")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 📸 Capture from Camera")
            camera_image = st.camera_input("Take a picture", key="camera_input")
            
            if camera_image is not None:
                st.session_state.camera_image = Image.open(camera_image)
                
                if st.button("🔍 Analyze Camera Image", key="analyze_camera"):
                    with st.spinner("Analyzing facial emotions..."):
                        result = analyze_image_emotion(st.session_state.camera_image)
                        if result:
                            display_results(result)
        
        with col2:
            st.markdown("##### 📤 Upload Image")
            uploaded_file = st.file_uploader("Choose an image...", 
                                            type=['jpg', 'jpeg', 'png'],
                                            key="image_upload")
            
            if uploaded_file is not None:
                st.session_state.uploaded_image = Image.open(uploaded_file)
                st.image(st.session_state.uploaded_image, caption="Uploaded Image", use_container_width=True)
                
                if st.button("🔍 Analyze Uploaded Image", key="analyze_upload"):
                    with st.spinner("Analyzing facial emotions..."):
                        result = analyze_image_emotion(st.session_state.uploaded_image)
                        if result:
                            display_results(result)
    
    # Voice Analysis Tab
    with tab3:
        st.markdown("#### Analyze Emotions from Voice")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 🎙️ Record Audio")
            duration = st.slider("Recording duration (seconds):", 3, 10, 5, key="duration_slider")
            
            if st.button("🔴 Start Recording", key="start_recording"):
                recording, sample_rate = record_audio(duration)
                
                # Save to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    write(tmp_file.name, sample_rate, recording)
                    st.session_state.recorded_audio = tmp_file.name
                    st.audio(tmp_file.name, format='audio/wav')
            
            if st.session_state.recorded_audio and st.button("🔍 Analyze Recording", key="analyze_recording"):
                with st.spinner("Analyzing voice emotions..."):
                    result = analyze_audio_emotion(st.session_state.recorded_audio)
                    if result:
                        display_results(result)
                    
                    # Clean up temp file
                    try:
                        os.unlink(st.session_state.recorded_audio)
                        st.session_state.recorded_audio = None
                    except:
                        pass
        
        with col2:
            st.markdown("##### 📤 Upload Audio")
            audio_file = st.file_uploader("Choose an audio file...", 
                                         type=['wav', 'mp3', 'ogg', 'm4a'],
                                         key="audio_upload")
            
            if audio_file is not None:
                st.audio(audio_file, format=f'audio/{audio_file.name.split(".")[-1]}')
                
                if st.button("🔍 Analyze Uploaded Audio", key="analyze_audio_upload"):
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{audio_file.name.split(".")[-1]}') as tmp_file:
                        tmp_file.write(audio_file.read())
                        temp_path = tmp_file.name
                    
                    with st.spinner("Analyzing voice emotions..."):
                        result = analyze_audio_emotion(temp_path)
                        if result:
                            display_results(result)
                    
                    # Clean up
                    try:
                        os.unlink(temp_path)
                    except:
                        pass

if __name__ == "__main__":
    main()
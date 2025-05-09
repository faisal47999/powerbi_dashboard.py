import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import tempfile
import os
from io import BytesIO
import base64
from scipy import signal
import time

# Set the page configuration
st.set_page_config(
    page_title="AI Voice Editor",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for the beautiful background and UI elements
st.markdown("""
<style>
    .main {
        background-image: linear-gradient(to right top, #051937, #004d7a, #008793, #00bf72, #a8eb12);
        background-attachment: fixed;
        color: white;
    }
    .sidebar .sidebar-content {
        background-image: linear-gradient(to bottom, #051937, #004d7a, #008793);
        color: white;
    }
    .stSlider > div > div > div {
        background-color: #00bf72;
    }
    .stButton>button {
        background-color: #a8eb12;
        color: #051937;
        font-weight: bold;
        border-radius: 20px;
        padding: 10px 20px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #00bf72;
        color: white;
    }
    .css-18e3th9 {
        padding-top: 1rem;
    }
    h1, h2, h3 {
        color: #a8eb12;
    }
    .stAlert {
        background-color: rgba(255, 255, 255, 0.1);
        border: 1px solid #a8eb12;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.title("🎙️ AI Voice Editor")
st.subheader("Transform your voice with AI-powered effects")

# Initialize session state variables
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None
if 'sample_rate' not in st.session_state:
    st.session_state.sample_rate = 22050
if 'processed_audio' not in st.session_state:
    st.session_state.processed_audio = None
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'current_effect' not in st.session_state:
    st.session_state.current_effect = None

# Function to get autoplay audio HTML
def get_audio_player_html(audio_data, sample_rate):
    audio_bytes = BytesIO()
    sf.write(audio_bytes, audio_data, sample_rate, format='WAV')
    audio_bytes.seek(0)
    b64 = base64.b64encode(audio_bytes.read()).decode()
    return f'<audio autoplay controls src="data:audio/wav;base64,{b64}"></audio>'

# Simulate AI processing with a progress bar
def simulate_ai_processing():
    st.session_state.processing = True
    progress_bar = st.progress(0)
    for i in range(100):
        time.sleep(0.02)  # Simulate processing time
        progress_bar.progress(i + 1)
    st.session_state.processing = False
    return True

# Audio processing functions
def apply_pitch_shift(audio_data, sample_rate, n_steps):
    return librosa.effects.pitch_shift(audio_data, sr=sample_rate, n_steps=n_steps)

def apply_time_stretch(audio_data, rate):
    return librosa.effects.time_stretch(audio_data, rate=rate)

def apply_reverb(audio_data, sample_rate, room_size=0.8):
    # Simulate reverb with a simple convolution
    reverb_length = int(sample_rate * room_size)
    impulse_response = np.exp(-np.linspace(0, 10, reverb_length))
    return signal.convolve(audio_data, impulse_response, mode='full')[:len(audio_data)]

def apply_robot_effect(audio_data, sample_rate):
    # Apply a simple robot effect by modulating with a carrier signal
    carrier = np.sin(2 * np.pi * 200 * np.arange(len(audio_data)) / sample_rate)
    modulated = audio_data * carrier
    return modulated

def apply_ai_voice_enhancement(audio_data):
    # Simulate AI enhancement with basic audio processing
    # In a real app, this would use a trained ML model
    enhanced = audio_data.copy()
    
    # Simulate processing
    simulate_ai_processing()
    
    # Apply a simple normalization as a placeholder for AI enhancement
    enhanced = enhanced / np.max(np.abs(enhanced))
    return enhanced

# Function to display audio waveform and spectrogram
def display_audio_visualization(audio_data, sample_rate):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    
    # Display waveform
    librosa.display.waveshow(audio_data, sr=sample_rate, ax=ax1)
    ax1.set_title('Waveform')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    
    # Display spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
    img = librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sample_rate, ax=ax2)
    fig.colorbar(img, ax=ax2, format="%+2.0f dB")
    ax2.set_title('Spectrogram')
    
    # Make the plot background transparent to match the Streamlit theme
    fig.patch.set_alpha(0.0)
    ax1.patch.set_alpha(0.3)
    ax2.patch.set_alpha(0.3)
    
    # Change text colors to be visible against dark background
    for ax in [ax1, ax2]:
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
    
    st.pyplot(fig)

# Sidebar for audio input and effects
with st.sidebar:
    st.header("Audio Input")
    
    # Option to upload audio file
    uploaded_file = st.file_uploader("Upload an audio file", type=['wav', 'mp3', 'ogg'])
    
    if uploaded_file is not None:
        st.success("File uploaded successfully!")
        
        # Convert the uploaded file to an audio array
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        audio_data, sample_rate = librosa.load(tmp_file_path, sr=None)
        os.unlink(tmp_file_path)  # Delete the temporary file
        
        st.session_state.audio_data = audio_data
        st.session_state.sample_rate = sample_rate
        st.session_state.processed_audio = None
    
    # Or record audio directly
    st.header("Or Record Audio")
    if st.button("Record Audio (10s)"):
        st.warning("This is a simulation. In a real app, this would access your microphone.")
        with st.spinner("Recording..."):
            # Simulate recording by generating a sample audio
            duration = 3  # seconds
            t = np.linspace(0, duration, int(22050 * duration), endpoint=False)
            # Generate a sample voice-like signal
            audio_data = 0.5 * np.sin(2 * np.pi * 200 * t) * np.exp(-t/2)
            # Add some noise to make it sound more natural
            audio_data += 0.1 * np.random.randn(len(audio_data))
            
            st.session_state.audio_data = audio_data
            st.session_state.sample_rate = 22050
            st.session_state.processed_audio = None
            
            st.success("Recording complete!")
    
    # Effects section
    if st.session_state.audio_data is not None:
        st.header("Voice Effects")
        
        effect_option = st.selectbox(
            "Select Effect",
            ["None", "Pitch Shift", "Time Stretch", "Reverb", "Robot Voice", "AI Enhancement"]
        )
        
        # Display sliders based on the selected effect
        if effect_option == "Pitch Shift":
            pitch_steps = st.slider("Pitch Shift (semitones)", -12.0, 12.0, 0.0, 0.5)
            if st.button("Apply Pitch Shift"):
                st.session_state.current_effect = "Pitch Shift"
                st.session_state.processed_audio = apply_pitch_shift(
                    st.session_state.audio_data, 
                    st.session_state.sample_rate, 
                    pitch_steps
                )
        
        elif effect_option == "Time Stretch":
            stretch_rate = st.slider("Time Stretch Factor", 0.5, 2.0, 1.0, 0.05)
            if st.button("Apply Time Stretch"):
                st.session_state.current_effect = "Time Stretch"
                st.session_state.processed_audio = apply_time_stretch(
                    st.session_state.audio_data, 
                    stretch_rate
                )
        
        elif effect_option == "Reverb":
            room_size = st.slider("Room Size", 0.1, 0.9, 0.5, 0.1)
            if st.button("Apply Reverb"):
                st.session_state.current_effect = "Reverb"
                st.session_state.processed_audio = apply_reverb(
                    st.session_state.audio_data, 
                    st.session_state.sample_rate, 
                    room_size
                )
        
        elif effect_option == "Robot Voice":
            if st.button("Apply Robot Effect"):
                st.session_state.current_effect = "Robot Voice"
                st.session_state.processed_audio = apply_robot_effect(
                    st.session_state.audio_data,
                    st.session_state.sample_rate
                )
        
        elif effect_option == "AI Enhancement":
            if st.button("Apply AI Enhancement"):
                st.session_state.current_effect = "AI Enhancement"
                st.session_state.processed_audio = apply_ai_voice_enhancement(
                    st.session_state.audio_data
                )
        
        # Reset button
        if st.button("Reset to Original"):
            st.session_state.processed_audio = None
            st.session_state.current_effect = None

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Display audio visualization
    if st.session_state.audio_data is not None:
        st.header("Audio Visualization")
        
        # Display the audio that should be visualized (original or processed)
        audio_to_display = st.session_state.processed_audio if st.session_state.processed_audio is not None else st.session_state.audio_data
        
        display_audio_visualization(audio_to_display, st.session_state.sample_rate)

with col2:
    # Audio playback section
    if st.session_state.audio_data is not None:
        st.header("Audio Playback")
        
        # Display original audio
        st.subheader("Original Audio")
        st.audio(st.session_state.audio_data, sample_rate=st.session_state.sample_rate)
        
        # Display processed audio if available
        if st.session_state.processed_audio is not None:
            st.subheader(f"Processed Audio ({st.session_state.current_effect})")
            st.audio(st.session_state.processed_audio, sample_rate=st.session_state.sample_rate)
            
            # Export button
            if st.button("Export Processed Audio"):
                # Create a download link for the processed audio
                buf = BytesIO()
                sf.write(buf, st.session_state.processed_audio, st.session_state.sample_rate, format='WAV')
                buf.seek(0)
                
                st.download_button(
                    label="Download WAV",
                    data=buf,
                    file_name="processed_voice.wav",
                    mime="audio/wav"
                )

# Footer
st.markdown("---")
with st.expander("About this AI Voice Editor"):
    st.markdown("""
    This is a simplified demonstration of an AI-powered voice editing application. In a production environment, this would include:
    
    - Real AI voice models for more sophisticated transformations
    - User authentication and project saving
    - More advanced audio processing capabilities
    - Multi-track editing
    - Collaboration features
    
    The current version demonstrates the UI and basic audio processing capabilities that could be integrated with more advanced AI models.
    """)

# Requirements section
with st.expander("Requirements to run this app"):
    st.code("""
    # requirements.txt
    streamlit==1.24.0
    numpy==1.24.3
    librosa==0.10.0
    matplotlib==3.7.1
    soundfile==0.12.1
    scipy==1.10.1
    """)

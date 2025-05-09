import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr
from scipy.signal import convolve, butter, lfilter
import matplotlib.pyplot as plt
import io
import uuid
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Custom CSS for beautiful design and slider visibility
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #1f77b4 0%, #ff7f0e 100%);
    color: white;
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
    font-family: 'Arial', sans-serif;
}
h1 {
    text-align: center;
    font-size: 2.5rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}
h3 {
    color: #333;
}
.card {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 12px;
    padding: 20px;
    margin: 15px 0;
    box-shadow: 0 6px 12px rgba(0,0,0,0.2);
}
.stButton>button {
    background-color: #ff7f0e;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px;
    font-size: 1rem;
    font-weight: bold;
    width: 100%;
}
.stButton>button:hover {
    background-color: #1f77b4;
}
.stSlider > div > div > div > div {
    background-color: #e6f3ff;
    border-radius: 6px;
    padding: 5px;
}
.stSlider label {
    font-weight: bold;
    color: #333;
    font-size: 1rem;
}
.stFileUploader {
    background: rgba(255, 255, 255, 0.9);
    border-radius: 8px;
    padding: 10px;
}
.footer {
    text-align: center;
    color: rgba(255, 255, 255, 0.8);
    margin-top: 30px;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# Function to load audio
def load_audio(file):
    try:
        audio, sr = librosa.load(file, sr=None)
        return audio, sr
    except Exception as e:
        st.error(f"Error loading audio: {str(e)}")
        return None, None

# Function to save audio
def save_audio(audio, sr):
    buffer = io.BytesIO()
    sf.write(buffer, audio, sr, format='wav')
    buffer.seek(0)
    return buffer

# Noise reduction
def reduce_noise(audio, sr, intensity=0.5):
    try:
        reduced_audio = nr.reduce_noise(y=audio, sr=sr, stationary=False, prop_decrease=intensity)
        return reduced_audio
    except Exception as e:
        st.error(f"Error in noise reduction: {str(e)}")
        return audio

# Pitch shift
def pitch_shift(audio, sr, n_steps):
    try:
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
    except Exception as e:
        st.error(f"Error in pitch shift: {str(e)}")
        return audio

# Tempo change
def change_tempo(audio, sr, rate):
    try:
        return librosa.effects.time_stretch(audio, rate=rate)
    except Exception as e:
        st.error(f"Error in tempo change: {str(e)}")
        return audio

# Reverb effect
def apply_reverb(audio, sr, reverb_intensity):
    try:
        decay = np.exp(-np.linspace(0, 5, int(sr * 0.5)))
        impulse_response = decay * reverb_intensity
        impulse_response = np.concatenate([impulse_response, np.zeros(len(audio) - len(impulse_response))])
        audio_reverb = convolve(audio, impulse_response, mode='same')
        return audio_reverb / np.max(np.abs(audio_reverb))
    except Exception as e:
        st.error(f"Error in reverb: {str(e)}")
        return audio

# Equalizer
def apply_equalizer(audio, sr, bass_gain=0.0, treble_gain=0.0):
    try:
        if bass_gain != 0:
            b, a = butter(4, 200 / (sr / 2), btype='low')
            audio = lfilter(b, a, audio) * (1 + bass_gain)
        if treble_gain != 0:
            b, a = butter(4, 2000 / (sr / 2), btype='high')
            audio = lfilter(b, a, audio) * (1 + treble_gain)
        return audio / np.max(np.abs(audio))
    except Exception as e:
        st.error(f"Error in equalizer: {str(e)}")
        return audio

# Plot waveform
def plot_waveform(audio, sr, title="Waveform"):
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(audio, sr=sr, color='#1f77b4')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    return buffer

# Streamlit App
st.title("🎙️ Voice Editor - Audio Studio")
st.markdown("""
<div class="card">
    <h3 style="text-align: center;">Welcome to Audio Studio</h3>
    <p style="color: #555; text-align: center;">
        Upload your naat or song and customize with professional effects!
    </p>
</div>
""", unsafe_allow_html=True)

# File uploader
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Upload Audio")
    uploaded_file = st.file_uploader("🎙️ Apni audio file upload karo (WAV format)", type=["wav"], label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    # Load audio
    audio, sr = load_audio(uploaded_file)
    if audio is None:
        st.stop()

    # Display original audio
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Original Audio")
        st.image(plot_waveform(audio, sr, "Original Audio Waveform"), use_column_width=True)
        st.audio(uploaded_file, format='audio/wav')
        st.markdown('</div>', unsafe_allow_html=True)

    # Effect controls with sliders
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Audio Effects")
        col1, col2 = st.columns(2)

        with col1:
            noise_reduction = st.checkbox("Noise Reduction", value=True)
            noise_intensity = st.slider("Noise Reduction Intensity", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
            pitch_steps = st.slider("Pitch Adjust (High/Low)", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)
            tempo_rate = st.slider("Tempo Adjust (Speed)", min_value=0.5, max_value=2.0, value=1.0, step=0.1)

        with col2:
            reverb_intensity = st.slider("Reverb (Echo)", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
            bass_gain = st.slider("Bass Boost", min_value=-0.5, max_value=0.5, value=0.0, step=0.1)
            treble_gain = st.slider("Treble Boost", min_value=-0.5, max_value=0.5, value=0.0, step=0.1)

        st.markdown('</div>', unsafe_allow_html=True)

    # Process button
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if st.button("🎵 Audio Process Karo", use_container_width=True):
            with st.spinner("Processing audio..."):
                processed_audio = audio.copy()

                if noise_reduction:
                    processed_audio = reduce_noise(processed_audio, sr, noise_intensity)
                if pitch_steps != 0:
                    processed_audio = pitch_shift(processed_audio, sr, pitch_steps)
                if tempo_rate != 1.0:
                    processed_audio = change_tempo(processed_audio, sr, tempo_rate)
                if reverb_intensity > 0:
                    processed_audio = apply_reverb(processed_audio, sr, reverb_intensity)
                if bass_gain != 0 or treble_gain != 0:
                    processed_audio = apply_equalizer(processed_audio, sr, bass_gain, treble_gain)

                output_buffer = save_audio(processed_audio, sr)

                st.markdown("### Processed Audio")
                st.image(plot_waveform(processed_audio, sr, "Processed Audio Waveform"), use_column_width=True)
                st.audio(output_buffer, format='audio/wav')
                st.download_button(
                    label="📥 Processed Audio Download Karo",
                    data=output_buffer,
                    file_name=f"processed_audio_{uuid.uuid4()}.wav",
                    mime="audio/wav",
                    use_container_width=True
                )
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">Made with ❤️ by xAI for audio enthusiasts!</div>', unsafe_allow_html=True)

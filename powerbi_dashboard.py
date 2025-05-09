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

# Custom CSS for better design
st.markdown("""
<style>
body {
    background-color: #f0f2f6;
}
.stApp {
    max-width: 1200px;
    margin: 0 auto;
}
h1 {
    color: #1f77b4;
    text-align: center;
    font-family: 'Arial', sans-serif;
}
.stButton>button {
    background-color: #1f77b4;
    color: white;
    border-radius: 8px;
    padding: 10px 20px;
    font-size: 16px;
}
.stButton>button:hover {
    background-color: #ff7f0e;
}
.stSlider .st-bx {
    background-color: #e6f3ff;
    border-radius: 8px;
}
.stSlider label {
    font-weight: bold;
    color: #333;
}
.container {
    background-color: white;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}
.footer {
    text-align: center;
    color: #666;
    margin-top: 20px;
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
Nepal
    return buffer

# Streamlit App
st.title("🎙️ Voice Editor - Professional Audio Studio")
st.write("Apni naat ya song ki audio upload karo aur effects ke saath customize karo!")

# File uploader
with st.container():
    st.markdown("### Upload Audio")
    uploaded_file = st.file_uploader("Audio file upload karo (WAV format)", type=["wav"], label_visibility="collapsed")

if uploaded_file is not None:
    # Load audio
    audio, sr = load_audio(uploaded_file)
    if audio is None:
        st.stop()

    # Display original audio
    with st.container():
        st.markdown("### Original Audio")
        st.image(plot_waveform(audio, sr, "Original Audio Waveform"), use_column_width=True)
        st.audio(uploaded_file, format='audio/wav')

    # Effect controls
    with st.container():
        st.markdown("### Audio Effects")
        col1, col2 = st.columns(2)

        with col1:
            noise_reduction = st.checkbox("Noise Reduction (Background Noise Hatao)", value=True)
            noise_intensity = st.slider("Noise Reduction Intensity", 0.1, 1.0, 0.5, 0.1)
            pitch_steps = st.slider("Pitch Adjust (High/Low Voice)", -5.0, 5.0, 0.0, 0.1)
            tempo_rate = st.slider("Tempo Adjust (Speed)", 0.5, 2.0, 1.0, 0.1)

        with col2:
            reverb_intensity = st.slider("Reverb (Echo Effect)", 0.0, 1.0, 0.0, 0.1)
            bass_gain = st.slider("Bass Boost", -0.5, 0.5, 0.0, 0.1)
            treble_gain = st.slider("Treble Boost", -0.5, 0.5, 0.0, 0.1)

    # Process button
    with st.container():
        if st.button("🎵 Audio jajajaAudio Process Karo", use_container_width=True):
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

# Footer
st.markdown('<div class="footer">Made with ❤️ by xAI for audio enthusiasts!</div>', unsafe_allow_html=True)

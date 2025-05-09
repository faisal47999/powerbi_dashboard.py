import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr
from scipy.signal import convolve, butter, lfilter
import matplotlib.pyplot as plt
import io
import os
import uuid
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Function to load audio
def load_audio(file):
    try:
        audio, sr = librosa.load(file, sr=None)
        return audio, sr
    except Exception as e:
        st.error(f"Error loading audio: {str(e)}")
        return None, None

# Function to save audio to bytes for download
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

# Equalizer (bass/treble boost)
def apply_equalizer(audio, sr, bass_gain=0.0, treble_gain=0.0):
    try:
        # Bass filter (low-pass, boost frequencies < 200 Hz)
        if bass_gain != 0:
            b, a = butter(4, 200 / (sr / 2), btype='low')
            audio = lfilter(b, a, audio) * (1 + bass_gain)
        
        # Treble filter (high-pass, boost frequencies > 2000 Hz)
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
    librosa.display.waveshow(audio, sr=sr)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    return buffer

# Streamlit App
st.title("Voice Editor - Professional Audio Processor")
st.write("Apni naat ya song ki audio upload karo aur effects apply karo!")

# File uploader
uploaded_file = st.file_uploader("Audio file upload karo (WAV format)", type=["wav"])

if uploaded_file is not None:
    # Load audio
    audio, sr = load_audio(uploaded_file)
    if audio is None:
        st.stop()

    # Display original waveform
    st.image(plot_waveform(audio, sr, "Original Audio Waveform"), use_column_width=True)
    st.audio(uploaded_file, format='audio/wav')
    st.write("Original Audio ⬆️")

    # Effect controls
    st.subheader("Audio Effects")
    col1, col2 = st.columns(2)

    with col1:
        noise_reduction = st.checkbox("Noise Reduction", value=True)
        noise_intensity = st.slider("Noise Reduction Intensity", 0.1, 1.0, 0.5, 0.1)
        pitch_steps = st.slider("Pitch Adjust (High/Low)", -5.0, 5.0, 0.0, 0.1)
        tempo_rate = st.slider("Tempo Adjust (Speed)", 0.5, 2.0, 1.0, 0.1)

    with col2:
        reverb_intensity = st.slider("Reverb (Echo)", 0.0, 1.0, 0.0, 0.1)
        bass_gain = st.slider("Bass Boost", -0.5, 0.5, 0.0, 0.1)
        treble_gain = st.slider("Treble Boost", -0.5, 0.5, 0.0, 0.1)

    # Process button
    if st.button("Audio Process Karo"):
        with st.spinner("Processing audio..."):
            processed_audio = audio.copy()

            # Apply noise reduction
            if noise_reduction:
                processed_audio = reduce_noise(processed_audio, sr, noise_intensity)

            # Apply pitch shift
            if pitch_steps != 0:
                processed_audio = pitch_shift(processed_audio, sr, pitch_steps)

            # Apply tempo change
            if tempo_rate != 1.0:
                processed_audio = change_tempo(processed_audio, sr, tempo_rate)

            # Apply reverb
            if reverb_intensity > 0:
                processed_audio = apply_reverb(processed_audio, sr, reverb_intensity)

            # Apply equalizer
            if bass_gain != 0 or treble_gain != 0:
                processed_audio = apply_equalizer(processed_audio, sr, bass_gain, treble_gain)

            # Save processed audio
            output_buffer = save_audio(processed_audio, sr)

            # Display processed waveform
            st.image(plot_waveform(processed_audio, sr, "Processed Audio Waveform"), use_column_width=True)
            st.audio(output_buffer, format='audio/wav')
            st.write("Processed Audio ⬆️")

            # Download button
            st.download_button(
                label="Processed Audio Download Karo",
                data=output_buffer,
                file_name=f"processed_audio_{uuid.uuid4()}.wav",
                mime="audio/wav"
            )

# Footer
st.markdown("---")
st.write("Made with ❤️ by xAI for audio enthusiasts!")

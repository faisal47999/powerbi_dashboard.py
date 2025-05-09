import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr
from scipy.signal import convolve
import io
import os
import uuid

# Function to load audio
def load_audio(file):
    audio, sr = librosa.load(file, sr=None)
    return audio, sr

# Function to save audio to bytes for download
def save_audio(audio, sr):
    buffer = io.BytesIO()
    sf.write(buffer, audio, sr, format='wav')
    buffer.seek(0)
    return buffer

# Noise reduction
def reduce_noise(audio, sr):
    reduced_audio = nr.reduce_noise(y=audio, sr=sr, stationary=False)
    return reduced_audio

# Pitch shift
def pitch_shift(audio, sr, n_steps):
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

# Tempo change
def change_tempo(audio, sr, rate):
    return librosa.effects.time_stretch(audio, rate=rate)

# Reverb effect (simple convolution with impulse response)
def apply_reverb(audio, sr, reverb_intensity):
    # Create a simple exponential decay impulse response
    decay = np.exp(-np.linspace(0, 5, int(sr * 0.5)))
    impulse_response = decay * reverb_intensity
    impulse_response = np.concatenate([impulse_response, np.zeros(len(audio) - len(impulse_response))])
    audio_reverb = convolve(audio, impulse_response, mode='same')
    return audio_reverb / np.max(np.abs(audio_reverb))

# Streamlit App
st.title("Voice Editor - Naat & Song Audio Processor")
st.write("Apni audio file upload karo aur effects apply karo!")

# File uploader
uploaded_file = st.file_uploader("Audio file upload karo (WAV format)", type=["wav"])

if uploaded_file is not None:
    # Load audio
    audio, sr = load_audio(uploaded_file)
    st.audio(uploaded_file, format='audio/wav')
    st.write("Original Audio ⬆️")

    # Noise reduction toggle
    noise_reduction = st.checkbox("Noise Reduction (Background noise hatao)", value=True)

    # Pitch shift slider
    pitch_steps = st.slider("Pitch Adjust (High/Low voice)", -5.0, 5.0, 0.0, 0.1)

    # Tempo slider
    tempo_rate = st.slider("Tempo Adjust (Speed badhao/kam karo)", 0.5, 2.0, 1.0, 0.1)

    # Reverb slider
    reverb_intensity = st.slider("Reverb (Echo effect)", 0.0, 1.0, 0.0, 0.1)

    # Process button
    if st.button("Audio Process Karo"):
        processed_audio = audio.copy()

        # Apply noise reduction
        if noise_reduction:
            with st.spinner("Noise hata rahe hain..."):
                processed_audio = reduce_noise(processed_audio, sr)

        # Apply pitch shift
        if pitch_steps != 0:
            with st.spinner("Pitch adjust kar rahe hain..."):
                processed_audio = pitch_shift(processed_audio, sr, pitch_steps)

        # Apply tempo change
        if tempo_rate != 1.0:
            with st.spinner("Tempo adjust kar rahe hain..."):
                processed_audio = change_tempo(processed_audio, sr, tempo_rate)

        # Apply reverb
        if reverb_intensity > 0:
            with st.spinner("Reverb add kar rahe hain..."):
                processed_audio = apply_reverb(processed_audio, sr, reverb_intensity)

        # Save processed audio
        output_buffer = save_audio(processed_audio, sr)

        # Play processed audio
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
st.write("Made with ❤️ by xAI for audio lovers!")

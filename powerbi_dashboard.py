import streamlit as st
import numpy as np
import soundfile as sf
import io
import noisereduce as nr
import librosa
import sounddevice as sd
from pydub import AudioSegment
from pydub.playback import play

# Function to record audio from mic
def record_audio(duration=5, fs=44100):
    st.info(f"Recording for {duration} seconds...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    return np.squeeze(recording), fs

# Function to apply noise reduction
def reduce_noise(audio, fs):
    reduced_noise = nr.reduce_noise(y=audio, sr=fs)
    return reduced_noise

# Function to change pitch
def change_pitch(audio, fs, n_steps):
    return librosa.effects.pitch_shift(audio, sr=fs, n_steps=n_steps)

# Function to change speed
def change_speed(audio, speed=1.0):
    return librosa.effects.time_stretch(audio, rate=speed)

# Convert numpy array to bytes for download
def audio_to_bytes(audio, fs):
    buffer = io.BytesIO()
    sf.write(buffer, audio, fs, format='wav')
    return buffer.getvalue()

st.title("Advanced Voice Editor Web App")

# Record audio
duration = st.slider("Recording Duration (seconds)", 1, 10, 5)
if st.button("Start Recording"):
    audio, fs = record_audio(duration)
    st.audio(audio_to_bytes(audio, fs), format="audio/wav")
    st.session_state['audio'] = audio
    st.session_state['fs'] = fs

if 'audio' in st.session_state:
    audio = st.session_state['audio']
    fs = st.session_state['fs']

    # Noise reduction toggle
    if st.checkbox("Apply Noise Reduction"):
        audio = reduce_noise(audio, fs)
        st.audio(audio_to_bytes(audio, fs), format="audio/wav")

    # Pitch adjustment slider (-12 to +12 semitones)
    pitch_shift = st.slider("Pitch Shift (semitones)", -12, 12, 0)
    if pitch_shift != 0:
        audio = change_pitch(audio, fs, pitch_shift)
        st.audio(audio_to_bytes(audio, fs), format="audio/wav")

    # Speed adjustment slider (0.5x to 2x)
    speed = st.slider("Speed", 0.5, 2.0, 1.0)
    if speed != 1.0:
        audio = change_speed(audio, speed)
        st.audio(audio_to_bytes(audio, fs), format="audio/wav")

    # Download edited audio
    edited_audio_bytes = audio_to_bytes(audio, fs)
    st.download_button(
        label="Download Edited Audio",
        data=edited_audio_bytes,
        file_name="edited_audio.wav",
        mime="audio/wav"
    )

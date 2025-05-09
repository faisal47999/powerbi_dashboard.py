import streamlit as st
import io

# Custom CSS for beautiful background and styling
st.markdown("""
<style>
body {
    margin: 0;
    font-family: 'Arial', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #1f77b4 0%, #ff7f0e 100%);
    color: white;
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}
h1 {
    text-align: center;
    font-size: 3rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}
.card {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 15px;
    padding: 20px;
    margin: 20px 0;
    box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    transition: transform 0.3s;
}
.card:hover {
    transform: translateY(-5px);
}
.stButton>button {
    background-color: #ff7f0e;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 24px;
    font-size: 1.1rem;
    font-weight: bold;
    width: 100%;
    transition: background-color 0.3s;
}
.stButton>button:hover {
    background-color: #1f77b4;
}
.stFileUploader {
    background: rgba(255, 255, 255, 0.9);
    border-radius: 10px;
    padding: 10px;
}
.footer {
    text-align: center;
    color: rgba(255, 255, 255, 0.8);
    margin-top: 40px;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# Streamlit App
st.title("🎵 Audio Studio - Beautiful Edition")

# Intro
st.markdown("""
<div class="card">
    <h3 style="color: #333; text-align: center;">Welcome to Audio Studio</h3>
    <p style="color: #555; text-align: center;">
        Upload your audio file and experience a beautifully designed interface!
    </p>
</div>
""", unsafe_allow_html=True)

# File uploader
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("🎙️ Apni audio file upload karo (WAV format)", type=["wav"], label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

# Audio playback
if uploaded_file is not None:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Your Audio")
    st.audio(uploaded_file, format='audio/wav')
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">Made with ❤️ by xAI for audio lovers!</div>', unsafe_allow_html=True)

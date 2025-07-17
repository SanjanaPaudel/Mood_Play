import streamlit as st
import pandas as pd
import joblib

# Load everything from the single pickle file
package = joblib.load('Knn_Mood_Play.pkl')

# Unpack model and encoders
knn = package['model']
mood_encoder = package['mood_encoder']
time_encoder = package['time_encoder']
activity_encoder = package['activity_encoder']

# Load dataset for lookup
data = pd.read_csv('MoodPlay.csv')

# Page configuration
st.set_page_config(page_title="Mood Music Recommender 🎧", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #e0f7fa 0%, #fce4ec 100%);
        font-family: 'Segoe UI', sans-serif;
    }
    .stButton>button {
        background-color: #ff4081;
        color: white;
        border-radius: 10px;
        font-weight: bold;
        padding: 0.5em 2em;
        transition: 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #f50057;
        transform: scale(1.05);
    }
    .song-card {
        background-color: #ffffffcc;
        padding: 1em;
        border-radius: 12px;
        margin-bottom: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("🎵 Mood-Based Music Recommender")
st.subheader("Get a personalized playlist that matches your vibe.")

# Sidebar inputs
st.sidebar.markdown("### 🧠 Tell us how you're feeling...")
mood = st.sidebar.selectbox("🌈 Mood", mood_encoder.classes_)
time = st.sidebar.selectbox("🕒 Time of Day", time_encoder.classes_)
activity = st.sidebar.selectbox("🎯 Activity", activity_encoder.classes_)
energy = st.sidebar.slider("⚡ Energy Level", 0.0, 1.0, 0.5)

# Predict button
if st.sidebar.button("🎧 Get My Playlist"):
    try:
        # Encode input
        encoded_input = [
            mood_encoder.transform([mood])[0],
            time_encoder.transform([time])[0],
            activity_encoder.transform([activity])[0],
            energy
        ]

        # Get 5 nearest neighbors
        distances, indices = knn.kneighbors([encoded_input], n_neighbors=5)
        playlist = data.iloc[indices[0]][['Song', 'Mood', 'Activity']]

        st.success("Here's your personalized playlist based on your vibe:")

        # Display each song in a styled card
        for i, row in playlist.iterrows():
            st.markdown(f"""
                <div class="song-card">
                    <h4>🎶 {i+1}. {row['Song']}</h4>
                    <p><b>Mood:</b> {row['Mood']}<br>
                    <b>Activity:</b> {row['Activity']}</p>
                </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Oops! Something went wrong: {e}")

# Footer
st.markdown("---")
st.caption("Built with 💖 using KNN | Designed by Sanjana ✨")

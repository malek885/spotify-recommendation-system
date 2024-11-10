import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import requests
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import base64

# Define Spotify API credentials
CLIENT_ID = '9dcdc1003acd46448ec4c9f7c3373626'
CLIENT_SECRET = '84a057714fe141e898fe26cfe3819db4'
# Load your dataset
dataset = pd.read_csv("dataset.csv")
# Preprocess: select relevant features and normalize them
features = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
scaler = MinMaxScaler()
dataset[features] = scaler.fit_transform(dataset[features])

# Streamlit app setup
st.title("Spotify Playlist Song Recommender")
st.sidebar.header("Playlist Input")

# Collect Playlist URL
playlist_url = st.sidebar.text_input("Enter Spotify Playlist URL:")
num_recommendations = st.sidebar.slider("Number of Recommendations:", 1, 10, 5)

if st.sidebar.button("Get Recommendations"):
    if not playlist_url:
        st.warning("Please enter a playlist URL.")
    else:
        def fetch_access_token():
            token_url = 'https://accounts.spotify.com/api/token'
            client_credentials = f"{CLIENT_ID}:{CLIENT_SECRET}"
            client_credentials_base64 = base64.b64encode(client_credentials.encode()).decode()
            headers = {'Authorization': f'Basic {client_credentials_base64}'}
            data = {'grant_type': 'client_credentials'}
            response = requests.post(token_url, data=data, headers=headers)
            if response.status_code == 200:
                return response.json()['access_token']
            else:
                st.error("Failed to authenticate with Spotify API.")
                return None

        # Extract playlist ID
        def extract_playlist_id(url):
            return url.split('/')[-1].split('?')[0]

        # Fetch playlist data
        def fetch_playlist_data(playlist_id, access_token):
            sp = spotipy.Spotify(auth=access_token)
            tracks_data = []
            try:
                playlist_tracks = sp.playlist_tracks(playlist_id)
                for track_info in playlist_tracks['items']:
                    track = track_info['track']

                    features_data = sp.audio_features(track['id'])[0]
                    
                    # Using list comprehension to retrieve only the necessary features
                    track_data = {
                        'track_name': track['name'],
                        'artists': ', '.join([artist['name'] for artist in track['artists']]),
                        'album_name': track['album']['name'],
                        'popularity': track['popularity'],
                        **{feature: features_data[feature] for feature in features}
                    }
                    tracks_data.append(track_data)

                playlist_df = pd.DataFrame(tracks_data)

                # Normalize the audio features in the playlist data
                playlist_df[features] = scaler.transform(playlist_df[features])
                return playlist_df
            except Exception as e:
                st.error(f"Error fetching playlist data: {e}")
                return pd.DataFrame()
        def fetch_album_image(track_name, artist_name, access_token):
            """Fetch the album cover image for a track."""
            base_url = "https://api.spotify.com/v1/search"
            headers = {"Authorization": f"Bearer {access_token}"}
            query = f"track:{track_name} artist:{artist_name}"
            params = {"q": query, "type": "track", "limit": 1}
            response = requests.get(base_url, headers=headers, params=params)
            if response.status_code == 200:
                data = response.json()
                if data['tracks']['items']:
                    return data['tracks']['items'][0]['album']['images'][0]['url']
            return None
        # Recommendation logic
        def recommend_songs(playlist_df, dataset, num_recommendations):
            features = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
                        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
            scaler = MinMaxScaler()
            dataset[features] = scaler.fit_transform(dataset[features])
            playlist_df[features] = scaler.transform(playlist_df[features])
            recommendations = []
            for _, song in playlist_df.iterrows():
                song_features = song[features].values.reshape(1, -1)
                dataset_features = dataset[features].values
                similarity = cosine_similarity(song_features, dataset_features)[0]
                song_recommendations = dataset.iloc[similarity.argsort()[-num_recommendations:][::-1]]
                recommendations.append(song_recommendations[['track_name', 'artists', 'album_name', 'popularity']])
            return pd.concat(recommendations).drop_duplicates().head(num_recommendations)

        

        # Execute the process
        access_token = fetch_access_token()
        if access_token:
            playlist_id = extract_playlist_id(playlist_url)
            playlist_df = fetch_playlist_data(playlist_id, access_token)

            if not playlist_df.empty:
                # Load your dataset
                dataset = pd.read_csv("dataset.csv")
                recommended_songs = recommend_songs(playlist_df, dataset, num_recommendations)
                st.subheader("Recommended Songs")
                recommended_songs = recommended_songs[~recommended_songs['track_name'].isin(playlist_df['track_name'])]
                # st.table(recommended_songs)
                # Custom CSS for styling
                for _, song in recommended_songs.iterrows():
                    album_image_url = fetch_album_image(song['track_name'], song['artists'], access_token)
                    if album_image_url:
                        st.image(album_image_url, width=50, caption=song['track_name'])
                        st.markdown(
                    f"""
                    <table class="styled-table">
                        <thead>
                            <tr>
                                <th>Tracks Name</th>
                                <th>Artists</th>
                                <th>Album Name</th>
                                <th>Popularity</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>{song['track_name']}</td>
                                <td>{song['artists']}</td>
                                <td>{song['album_name']}</td>
                                <td>{song['popularity']}</td>
                            </tr>
                        </tbody>
                    </table>
                    """,
                    unsafe_allow_html=True,
                )
                        st.markdown("<hr>", unsafe_allow_html=True)
            else:
                st.error("Failed to fetch playlist data.")
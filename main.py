import spotipy
from spotipy.oauth2 import SpotifyOAuth

from sentence_transformers import SentenceTransformer, util

import logging

logging.basicConfig(level=logging.INFO)

spotipy = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id="",
                                                    client_secret="",
                                                    redirect_uri="http://localhost:1234",
                                                    scope="playlist-modify-private"))
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

all_playlists = spotipy.current_user_playlists()['items']

season_indicator = ['spring', 'summer', 'fall', 'winter']
seasonal_playlists = [playlist for playlist in all_playlists
                      if any(indicator in playlist['name'].lower() for indicator in season_indicator)]

main_playlist_names = ['relaxing, chilling and vibing', 'Rock, alternative and between',
                        'Hip Hop and rap', 'Danceble pop bops']
main_playlists = [playlist for playlist in all_playlists
                  if playlist['name'] in main_playlist_names]

for playlist in seasonal_playlists:
    logging.info(f'Sorting seasonal playlist: {playlist['name']}')
    logging.info(f'Getting tracks ...')

    tracks = spotipy.playlist_items(playlist['id'], limit=100)['items']

    for track in tracks:
        track = track['track']

        artist = spotipy.artist(track['artists'][0]['id'])
        genres = artist['genres']

        genres = ' '.join(genres)
        genre_embedding = model.encode(genres)
        playlist_embeddings = model.encode(main_playlist_names)

        similarities = util.cos_sim(genre_embedding, playlist_embeddings)

        most_similar_idx = similarities.argmax()
        most_similar_text = main_playlist_names[most_similar_idx]

        logging.info(f'Sorting track: {track['name']} into playlist: {most_similar_text}')
# spotify-playlist-autosorter
Basis for automatically sorting songs from seasonal playlists into genre playlists.

Did not further develop this, because Spotify API does not provide genres on a per song basis and artist genres are too noisy with respect to potential song genres to work as basis for assigning a genre playlist. Assigns the playlist based on embedding similarity of artist genres and playlist names.
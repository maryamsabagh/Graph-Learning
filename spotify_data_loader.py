import json
import numpy as np
import pickle
import random
import os
from pathlib import Path as Data_Path
from basic_types import JSONFile
from tqdm import tqdm

def playlist_track_JSON_loader(JSONs):
  playlist_data = {}
  playlists = []
  tracks = []
  artists = []
  playlist_track_edges = []
  track_artist_edges = []
  # build list of all unique playlists, tracks, and artists 
  for json_file in tqdm(JSONs):
    playlists += [p.name for p in json_file.playlists.values()]
    tracks += [track.uri for playlist in json_file.playlists.values() for track in list(playlist.tracks.values())]
    artists += [track.artist_uri for playlist in json_file.playlists.values() for track in list(playlist.tracks.values())]
    
    playlist_track_edges += [(playlist.name, track.uri) for playlist in json_file.playlists.values() for track in list(playlist.tracks.values())]
    track_artist_edges += [(track.uri, track.artist_uri) for playlist in json_file.playlists.values() for track in list(playlist.tracks.values())]

    playlist_data = playlist_data | json_file.playlists
  return playlists, tracks, artists, playlist_track_edges, track_artist_edges, playlist_data

def spotify_data_loader(N_FILES_TO_USE, DATA_DIR):
  # set the seed for reproducibility
    seed = 224
    np.random.seed(seed)
    random.seed(seed)
    file_names = sorted([f for f in os.listdir(DATA_DIR) 
                            if os.path.isfile(os.path.join(DATA_DIR, f)) and f.endswith('.json')])

    file_names_to_use = file_names[:N_FILES_TO_USE]

    n_playlists = 0

    # load each json file, and store it in a list of files
    JSONs = []
    for file_name in tqdm(file_names_to_use, desc='Files processed: ', unit='files', total=len(file_names_to_use)):
        json_file = JSONFile(DATA_DIR, file_name, n_playlists)
        json_file.process_file()
        n_playlists += len(json_file.playlists)
        JSONs.append(json_file)

    playlists, tracks, artists, playlist_track, track_artist, playlist_data = playlist_track_JSON_loader(JSONs)
    return playlists, tracks, artists, playlist_track, track_artist, playlist_data


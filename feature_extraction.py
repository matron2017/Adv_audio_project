import os
import librosa
import librosa.display
import numpy as np
import pandas as pd
import pickle

# Function to extract mel-spectrogram from audio file with adjustable length
def extract_mel_spectrogram(audio_file, fixed_duration=5, sr=22050, n_mels=128):
    try:
        # Load audio file with fixed duration
        audio, sr = librosa.load(audio_file, sr=sr, duration=fixed_duration)

        # Calculate mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)

        # Convert to decibels
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        #print(f"Shape of spectrogram at extracting: {np.shape(mel_spectrogram)}")

        return mel_spectrogram_db

    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        return None

# Path to your metadata file
metadata_file = "Data/meta.csv"

# Read the metadata file using pandas
metadata = pd.read_csv(metadata_file, sep='\t')

# Output directory for saving mel-spectrograms
output_directory = "output_mel_spectrograms/"
os.makedirs(output_directory, exist_ok=True)

# Adjustable segment length in seconds
segment_length = 5
n_mels = 128

# Loop through each row in the metadata
count = 0
for index, row in metadata.iterrows():
    audio_file = row['filename']
    scene_label = row['scene_label']
    identifier = row['identifier']
    
    # Extract mel-spectrogram for each segment
    mel_spectrogram = extract_mel_spectrogram("Data/"+audio_file, fixed_duration=segment_length,n_mels=n_mels)
    #print(f"Shape of spectrograms: {np.shape(mel_spectrogram)}")
    #print(f"Shape of spectrogram at saving: {np.shape(mel_spectrogram)}")
    #Check correct shape
    if np.shape(mel_spectrogram) == (n_mels, 216):
        output_filename = f"{output_directory}{scene_label}_{count}.pkl"
        data_to_save = {'mel_spectrogram': mel_spectrogram, 'label': scene_label}
        with open(output_filename, 'wb') as f:
            pickle.dump(data_to_save, f)
            count += 1
            #print(f"Mel-spectrogram saved for {audio_file} as {output_filename}")
    else:
        print(f"Mel-spectrogram not saved for {audio_file}")
print(f"Total mel-spectrograms saved: {count}")
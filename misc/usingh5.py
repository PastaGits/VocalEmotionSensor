import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import librosa
import librosa.display
import IPython.display as ipd
import tensorflow as tf
import h5py
pwd = os.getcwd()
# load ravdess.npy , savee.npy and tess.npy
ravdess_data = np.load('ravdess.npy')
savee_data = np.load('savee.npy')
tess_data = np.load('tess.npy')
crema_data = np.load('crema.npy')

# ravdess_data.shape, savee_data.shape, tess_data.shape
all_data = np.vstack((ravdess_data, savee_data, tess_data, crema_data))
df = pd.DataFrame(all_data, columns=[
                  'label', 'gender', 'pathname', 'filename'])

# save all data into npy file
# np.save("all_datasets.npy", df)


def plot_spec(y, sr, hop_size, y_axis):
    plt.figure(figsize=(10, 7))
    librosa.display.specshow(
        y, sr=sr, hop_length=hop_size, x_axis='time', y_axis=y_axis)
    plt.colorbar(format='%+2.0f dB')


# load wav files using the 'pathname' and 'filename' columns of all_data with librosa
frame_size = 4096  # in samples
hop_size = 512  # in samples
temporal_chunk_size = 42  # number of temporal bins per sample
mel_bands = 128  # number of mel bands
silence_threshold = 40  # in  relative to peak dB
in_dB = True  # convert to dB
mfcc_coefficients = 12  # number of MFCC coefficients

# total temporal bins is total_samples/hop_size
data_cols = ['stft_data', 'mel_data', 'mfcc_data']
data = []
f = h5py.File('dataset_1.hdf5', 'a')


def add_to_dataset(temporal_chunks):
    # np.array([data.append(c) for c in temporal_chunks], dtype=np.float32)
    # for index, data_name in enumerate(data_cols)
    #     new_data = np.array([temporal_chunks[i][index] for i in range(len(temporal_chunks))])
    #     if not f.keys().__contains__(data_name):
    #         f.create_dataset(data_name, data=new_data,
    #                      compression="gzip", chunks=True, maxshape=(None, (frame_size/2)+1, temporal_chunk_size))
    #     if data_name != 'gender' or data_name!='label':
    #         f[data_name].resize((f[data_name].shape[0] + new_data.shape[0]), axis=0)
    #         f[data_name][-new_data.shape[0]:] = new_data
    
    n_chunks = len(temporal_chunks)
    new_stft_data = np.array([temporal_chunks[i][0] for i in range(n_chunks)])
    new_mel_data = np.array([temporal_chunks[i][1] for i in range(n_chunks)])
    new_mfcc_data = np.array([temporal_chunks[i][2] for i in range(n_chunks)])
    new_gender_data = np.array([temporal_chunks[i][3]
                               for i in range(n_chunks)]).reshape(n_chunks, 1).astype('S')
    new_label_data = np.array([temporal_chunks[i][4]
                              for i in range(n_chunks)]).reshape(n_chunks, 1).astype('S')
    if len(f.keys()) == 0:
        # create separate datasets for each col
        f.create_dataset('stft', data=new_stft_data,
                         compression="gzip", chunks=True, maxshape=(None, new_stft_data.shape[1], temporal_chunk_size))
        f.create_dataset('mel_spec', data=new_mel_data, compression="gzip",
                         chunks=True, maxshape=(None, mel_bands, temporal_chunk_size))
        f.create_dataset('mfcc', data=new_mfcc_data,
                         compression="gzip", chunks=True, maxshape=(None, mfcc_coefficients, temporal_chunk_size))
        f.create_dataset('gender', data=new_gender_data,
                         compression="gzip", chunks=True, maxshape=(None, 1))
        f.create_dataset('label', data=new_label_data,
                         compression="gzip", chunks=True, maxshape=(None, 1))
        return

    f['stft'].resize((f['stft'].shape[0] + new_stft_data.shape[0]), axis=0)
    f['stft'][-new_stft_data.shape[0]:] = new_stft_data

    f['mel_spec'].resize(
        (f['mel_spec'].shape[0] + new_mel_data.shape[0]), axis=0)
    f['mel_spec'][-new_mel_data.shape[0]:] = new_mel_data

    f['mfcc'].resize((f['mfcc'].shape[0] + new_mfcc_data.shape[0]), axis=0)
    f['mfcc'][-new_mfcc_data.shape[0]:] = new_mfcc_data

    f['gender'].resize(
        (f['gender'].shape[0] + new_gender_data.shape[0]), axis=0)
    f['gender'][-new_gender_data.shape[0]:] = new_gender_data

    f['label'].resize((f['label'].shape[0] + new_label_data.shape[0]), axis=0)
    f['label'][-new_label_data.shape[0]:] = new_label_data
for sample_index in range(all_data.shape[0]):
    temporal_chunks = []

    pathname = df['pathname'][sample_index]
    filename = df['filename'][sample_index]

    wav, sr = librosa.load(pwd + pathname + filename)
    trimmed_wav, _ = librosa.effects.trim(wav, top_db=silence_threshold)

    if sr != 22050:
        raise ValueError("Sample rate is not 22050Hz")

    # extract audio features for the audio file
    S_audio = librosa.stft(trimmed_wav, n_fft=frame_size, hop_length=hop_size)
    y_audio = np.abs(S_audio)

    mel_spec = librosa.feature.melspectrogram(
        S=y_audio, sr=sr, n_fft=frame_size, hop_length=hop_size, n_mels=mel_bands)

    mfccs = librosa.feature.mfcc(S=librosa.power_to_db(
        mel_spec), sr=sr, n_mfcc=mfcc_coefficients)

    if in_dB:
        y_audio = librosa.power_to_db(y_audio, ref=np.max)
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    # split y into chunks of size temporal_chunk_size.
    _y = (y_audio).T
    _mel_spec = (mel_spec).T
    _mfccs = (mfccs).T

    split_indices = np.unique([(i, len(_y) - temporal_chunk_size)[int(
        i + temporal_chunk_size >= len(_y))]for i in range(0, len(_y), temporal_chunk_size)])

    # to include mel-spec add `_mel_spec[i:i+temporal_chunk_size].T` in the list
    [temporal_chunks.append([
        np.array(_y[i:i+temporal_chunk_size].T, dtype=np.float32),
        np.array(_mel_spec[i:i+temporal_chunk_size].T, dtype=np.float32),
        np.array(_mfccs[i:i+temporal_chunk_size].T, dtype=np.float32),
        df['gender'][sample_index],
        df['label'][sample_index]])
     for i in split_indices]

    add_to_dataset(temporal_chunks)

f.attrs['sample_rate'] = sr
f.attrs['window_size'] = frame_size
f.attrs['hop_size'] = hop_size
f.close()
# MFCC features with numpy only (no librosa), for the live demo on the laptop.
#
# It does the same as librosa.feature.mfcc in the notebook:
#   n_mfcc=13, n_fft=512, win_length=480, hop_length=320, n_mels=40, center=False
# This way the laptop only needs numpy for the live demo, not librosa. To check that it really gives
# the same numbers as the notebook, run:   python tools/kws_features.py
# It compares the result with the features in firmware/kws_esp32/test_samples.h (made with librosa).

import json
import re
import sys
import wave
from pathlib import Path

import numpy as np

SR = 16000
CLIP_SAMPLES = SR
N_FFT = 512
WIN_LENGTH = 480
HOP_LENGTH = 320
N_MELS = 40
N_MFCC = 13
N_FRAMES = 1 + (CLIP_SAMPLES - N_FFT) // HOP_LENGTH   # 49
KEYWORDS = ['yes', 'no', 'left', 'right']
CLASSES = KEYWORDS + ['silence', 'unknown']           # same order as the model output


# ---- mel filterbank (librosa uses the "slaney" mel scale by default)
def hz_to_mel(f):
    f = np.asarray(f, dtype=np.float64)
    mel = f / (200.0 / 3)
    above = f >= 1000.0          # above 1 kHz the scale is logarithmic
    mel[above] = 15.0 + np.log(f[above] / 1000.0) / (np.log(6.4) / 27.0)
    return mel


def mel_to_hz(m):
    m = np.asarray(m, dtype=np.float64)
    f = m * (200.0 / 3)
    above = m >= 15.0
    f[above] = 1000.0 * np.exp(np.log(6.4) / 27.0 * (m[above] - 15.0))
    return f


def make_mel_filterbank():
    fft_freqs = np.fft.rfftfreq(N_FFT, 1.0 / SR)
    mel_points = np.linspace(hz_to_mel([0.0])[0], hz_to_mel([SR / 2])[0], N_MELS + 2)
    hz_points = mel_to_hz(mel_points)
    fb = np.zeros((N_MELS, len(fft_freqs)))
    for i in range(N_MELS):
        left, center, right = hz_points[i], hz_points[i + 1], hz_points[i + 2]
        rising = (fft_freqs - left) / (center - left)
        falling = (right - fft_freqs) / (right - center)
        fb[i] = np.maximum(0, np.minimum(rising, falling))
        fb[i] *= 2.0 / (right - left)   # "slaney" normalization
    return fb


def make_dct_matrix():
    # DCT type II with norm='ortho', only the first 13 rows
    n = np.arange(N_MELS)
    dct = np.zeros((N_MFCC, N_MELS))
    for k in range(N_MFCC):
        dct[k] = np.sqrt(2.0 / N_MELS) * np.cos(np.pi * k * (2 * n + 1) / (2 * N_MELS))
    dct[0] /= np.sqrt(2.0)
    return dct


# hann window with 480 samples, padded with zeros to 512 (librosa puts it in the middle)
WINDOW = np.zeros(N_FFT)
pad = (N_FFT - WIN_LENGTH) // 2
WINDOW[pad:pad + WIN_LENGTH] = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(WIN_LENGTH) / WIN_LENGTH)
MEL_FB = make_mel_filterbank()
DCT = make_dct_matrix()


# ---- same functions as in the notebook
def load_wav(path):
    # 16-bit wav -> float32 between -1 and 1
    with wave.open(str(path)) as w:
        if w.getframerate() != SR or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(f'{path}: expected 16 kHz, mono, 16-bit')
        data = w.readframes(w.getnframes())
    return np.frombuffer(data, dtype='<i2').astype(np.float32) / 32768.0


def best_window_start(y):
    c = np.concatenate([[0.0], np.cumsum(y.astype(np.float64) ** 2)])
    energy = c[CLIP_SAMPLES:] - c[:-CLIP_SAMPLES]
    good = np.where(energy >= 0.99 * energy.max())[0]
    return int((good[0] + good[-1]) // 2)


def fit_to_1s(y, mode='energy'):
    if len(y) <= CLIP_SAMPLES:
        return np.pad(y, (0, CLIP_SAMPLES - len(y)))
    start = best_window_start(y) if mode == 'energy' else 0
    return y[start:start + CLIP_SAMPLES]


def mfcc_features(y):
    # 1 s of audio -> (49, 13) MFCC matrix
    y = np.asarray(y, dtype=np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(y, N_FFT)[::HOP_LENGTH]
    power = np.abs(np.fft.rfft(frames * WINDOW, axis=1)) ** 2
    mel = power @ MEL_FB.T
    db = 10.0 * np.log10(np.maximum(1e-10, mel))
    db = np.maximum(db, db.max() - 80.0)   # librosa's top_db=80
    return (db @ DCT.T).astype(np.float32)


# ---- check against the features from the notebook
def check():
    root = Path(__file__).resolve().parents[1]
    samples_h = (root / 'firmware/kws_esp32/test_samples.h').read_text(encoding='utf-8')
    config_h = (root / 'firmware/kws_esp32/kws_config.h').read_text(encoding='utf-8')

    def floats(text):
        return np.array([float(v[:-1]) for v in re.findall(r'-?[\d.]+(?:e[-+]?\d+)?f', text)], np.float32)

    names = re.findall(r'"(\w+)"', samples_h.split('kTestNames', 1)[1].split('};', 1)[0])
    blocks = re.findall(r'\{([^{}]*)\}', samples_h.split('kTestFeatures', 1)[1])
    mean = floats(config_h.split('kFeatureMean', 1)[1].split('};', 1)[0])
    std = floats(config_h.split('kFeatureStd', 1)[1].split('};', 1)[0])
    metrics = json.loads((root / 'results/metrics.json').read_text(encoding='utf-8'))
    scale, zero = metrics['results']['cnn']['input_quant']

    def to_int8(x):
        return np.clip(np.rint((x - mean) / std / np.float32(scale)) + zero, -128, 127)

    biggest_diff = 0.0
    different = 0
    total = 0
    for name, block in zip(names, blocks):
        expected = floats(block).reshape(N_FRAMES, N_MFCC)
        wav = root / 'data/own_recordings' / f'{name}.wav'
        audio = load_wav(wav) if wav.exists() else np.zeros(CLIP_SAMPLES, np.float32)   # 'digital_silence'
        mine = mfcc_features(fit_to_1s(audio))
        biggest_diff = max(biggest_diff, float((np.abs(mine - expected) / std).max()))
        different += int((to_int8(mine) != to_int8(expected)).sum())
        total += expected.size
    print(f'{len(names)} clips compared with the librosa features from the notebook')
    print(f'biggest difference: {biggest_diff:.2e} standard deviations')
    print(f'int8 model inputs that are different: {different} of {total}')
    return different == 0


if __name__ == '__main__':
    sys.exit(0 if check() else 1)

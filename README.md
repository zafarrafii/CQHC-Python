# CQT-SEC-Python

Constant-Q transform spectral envelope coefficients (CQT-SEC), a timbre feature designed for music signals.

Files:
- [`cqtsec.py`](#cqtsecpy): Python module with the CQT-SEC and other related functions.
- [`examples.ipynb`](#examplesipynb): Jupyter notebook with some examples.
- [`bass_acoustic_000-036-075.wav`](#bass_acoustic_000-036-075wav): audio file used for the examples.

## cqtsec.py

This Python module implements the CQT-SEC and other related function.

Simply copy the file `cqtsec.py` in your working directory and you are good to go. Make sure you have Python 3, NumPy, and SciPy installed.

Functions:
- [`mfcc`](#mfcc) - Compute the mel-frequency cepstral coefficients (MFCCs) using librosa.
- [`cqt`](#cqt) - Compute the magnitude constant-Q transform (CQT) spectrogram using librosa.
- [`cqtdeconv`](#cqtdeconv) - Compute the pitch-independent spectral envelope and the energy-normalized pitch component from the CQT spectrogram.
- [`cqtsec`](#cqtsec) - Compute the CQT-SECs.


### mfcc

Compute the mel-frequency cepstral coefficients (MFCCs) using librosa.

```
audio_mfcc = mfcc(audio_signal, sampling_frequency, window_length, step_length, number_coefficients)
    
Inputs:
    audio_signal: audio signal (number_samples,)
    sampling_frequency: sampling frequency in Hz
    window_length: window length in samples
    step_length: step length in samples
    number_coefficients: number of MFCCs (default: 20 coefficients)
Output:
    audio_mfcc: audio MFCCs (number_coefficients, number_frames)
```

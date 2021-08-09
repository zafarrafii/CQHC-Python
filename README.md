# CQT-SEC-Python

Constant-Q transform spectral envelope coefficients (CQT-SEC), a timbre feature designed for music signals.

Files:
- [`cqtsec.py`](#cqtsecpy): Python module with the CQT-SEC and other related functions.
- [`tests.ipynb`](#testsipynb): Jupyter notebook with some tests.
- [`examples.ipynb`](#examplesipynb): Jupyter notebook with some examples.
- [`bass_acoustic_000-036-075.wav`](#bass_acoustic_000-036-075wav): audio file used for the tests and examples.

## cqtsec.py

This Python module implements the CQT-SEC and other related functions.

Simply copy the file `cqtsec.py` in your working directory and you are good to go. Make sure you have Python 3 and NumPy installed.

Functions:
- [`mfcc`](#mfcc) - Compute the mel-frequency cepstral coefficients (MFCCs) using librosa.
- [`cqt`](#cqt) - Compute the magnitude constant-Q transform (CQT) spectrogram using librosa.
- [`cqtdeconv`](#cqtdeconv) - Compute the pitch-independent spectral envelope and the energy-normalized pitch component from the CQT spectrogram.
- [`cqtsec`](#cqtsec) - Compute the CQT spectral envelope coefficients (CQT-SEC).

See also:
- [Zaf-Python](https://github.com/zafarrafii/Zaf-Python): Zafar's Audio Functions in Python for audio signal analysis.

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

### cqt

Compute the magnitude constant-Q transform (CQT) spectrogram using librosa.

```
cqt_spectrogram = cqt(audio_signal, sampling_frequency, step_length, minimum_frequency, octave_resolution)
    
Inputs:
    audio_signal: audio signal (number_samples,)
    sampling_frequency: sampling frequency in Hz
    step_length: step length in samples
    minimum_frequency: minimum frequency in Hz (default: 32.70 Hz = C1)
    octave_resolution: number of frequency channels per octave (default: 12 frequency channels per octave)
Output:
    cqt_spectrogram: magnitude CQT spectrogram (number_frequencies, number_frames)
```


#### Example:
```

```


### cqtdeconv

Compute the pitch-independent spectral envelope and the energy-normalized pitch component from the constant-Q transform (CQT) spectrogram.

```
cqt_envelope, cqt_pitch = cqtdeconv(cqt_spectrogram)
    
Inputs:
    cqt_spectrogram: CQT spectrogram (number_frequencies, number_frames)
Output:
    cqt_envelope: pitch-independent spectral envelope (number_frequencies, number_frames)
    cqt_pitch: energy-normalized pitch component (number_frequencies, number_frames)
```

#### Example:
```

```

### cqtsec

Compute the constant-Q transform (CQT) spectral envelope coefficients (CQT-SEC).

```
cqt_sec = cqtdeconv(audio_signal, sampling_frequency, step_length, minimum_frequency, octave_resolution, number_coefficients)
    
Inputs:
    cqt_spectrogram: CQT spectrogram (number_frequencies, number_frames)
Output:
    cqt_envelope: pitch-independent spectral envelope (number_frequencies, number_frames)
    cqt_pitch: energy-normalized pitch component (number_frequencies, number_frames)
```

#### Example:
```

```

## tests.ipynb

This Jupyter notebook shows some tests.

See [Jupyter notebook viewer](https://nbviewer.jupyter.org/github/zafarrafii/CQT-SEC-Python/blob/master/tests.ipynb).


## examples.ipynb

This Jupyter notebook shows some examples.

See [Jupyter notebook viewer](https://nbviewer.jupyter.org/github/zafarrafii/CQT-SEC-Python/blob/master/examples.ipynb).


## bass_acoustic_000-036-075.wav

4 second musical note of an acoustic bass playing C2 (65.41 Hz), from the [NSynth dataset](https://magenta.tensorflow.org/datasets/nsynth).


# References

- Brian McFee, Collin Raffel, Dawen Liang, Daniel PW Ellis, Matt McVicar, Eric Battenberg, and Oriol Nieto, "librosa: Audio and music signal analysis in python," 14th Python in Science Conference, Austin, TX, USA, July 6-12 2015 [[URL](https://arxiv.org/abs/1704.01279)]

- Jesse Engel, Cinjon Resnick, Adam Roberts, Sander Dieleman, Douglas Eck, Karen Simonyan, and Mohammad Norouzi, "Neural audio synthesis of musical notes with WaveNet
autoencoders," *34th International Conference on Machine Learning*, Sydney, NSW, Australia, August 6-11 2017 [[URL](https://librosa.org/doc/latest/index.html#)]


# Author

- Zafar Rafii
- zafarrafii@gmail.com
- http://zafarrafii.com/
- [CV](http://zafarrafii.com/Zafar%20Rafii%20-%20C.V..pdf)
- [GitHub](https://github.com/zafarrafii)
- [LinkedIn](https://www.linkedin.com/in/zafarrafii/)
- [Google Scholar](https://scholar.google.com/citations?user=8wbS2EsAAAAJ&hl=en)

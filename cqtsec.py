"""
This Python module implements the constant-Q transform spectral envelope coefficients (CQT-SEC) and other related functions.

Functions:
    mfcc - Compute the mel-frequency cepstral coefficients (MFCCs) using librosa.
    cqt - Compute the magnitude constant-Q transform (CQT) spectrogram using librosa.
    cqtdeconv - Compute the pitch-independent spectral envelope and the energy-normalized pitch component from the CQT spectrogram.
    cqtsec - Compute the constant-Q transform (CQT) spectral envelope coefficients (CQT-SEC).

Author:
    Zafar Rafii
    zafarrafii@gmail.com
    http://zafarrafii.com
    https://github.com/zafarrafii
    https://www.linkedin.com/in/zafarrafii/
    06/16/21
"""

import numpy as np
import librosa


def mfcc(
    audio_signal, sampling_frequency, window_length, step_length, number_coefficients=20
):
    """
    Compute the mel-frequency cepstral coefficients (MFCCs) using librosa.

    Inputs:
        audio_signal: audio signal (number_samples,)
        sampling_frequency: sampling frequency in Hz
        window_length: window length in samples
        step_length: step length in samples
        number_coefficients: number of MFCCs (default: 20 coefficients)
    Output:
        audio_mfcc: audio MFCCs (number_coefficients, number_frames)
    """

    # Compute the MFCCs using librosa's mfcc
    audio_mfcc = librosa.feature.mfcc(
        y=audio_signal,
        sr=sampling_frequency,
        n_fft=window_length,
        hop_length=step_length,
        n_mfcc=number_coefficients,
    )

    return audio_mfcc


def cqt(
    audio_signal,
    sampling_frequency,
    step_length,
    minimum_frequency=32.70,
    octave_resolution=12,
):
    """
    Compute the magnitude constant-Q transform (CQT) spectrogram using librosa.

    Inputs:
        audio_signal: audio signal (number_samples,)
        sampling_frequency: sampling frequency in Hz
        step_length: step length in samples
        minimum_frequency: minimum frequency in Hz (default: 32.70 Hz = C1)
        octave_resolution: number of frequency channels per octave (default: 12 frequency channels per octave)
    Output:
        cqt_spectrogram: magnitude CQT spectrogram (number_frequencies, number_frames)
    """

    # Derive the number of frequency channels
    maximum_frequency = sampling_frequency / 2
    number_frequencies = round(
        octave_resolution * np.log2(maximum_frequency / minimum_frequency)
    )

    # Compute the magnitude CQT spectrogram using librosa
    cqt_spectrogram = np.abs(
        librosa.cqt(
            audio_signal,
            sr=sampling_frequency,
            hop_length=step_length,
            fmin=minimum_frequency,
            bins_per_octave=octave_resolution,
            n_bins=number_frequencies,
        )
    )

    return cqt_spectrogram


def cqtdeconv(cqt_spectrogram):
    """
    Compute the pitch-independent spectral envelope and the energy-normalized pitch component from the constant-Q transform (CQT) spectrogram.

    Inputs:
        cqt_spectrogram: CQT spectrogram (number_frequencies, number_frames)
    Output:
        cqt_envelope: pitch-independent spectral envelope (number_frequencies, number_frames)
        cqt_pitch: energy-normalized pitch component (number_frequencies, number_frames)
    """

    # Get the number of frequency channels
    number_frequencies = np.shape(cqt_spectrogram)[0]

    # Compute the Fourier transform of every frame and their magnitude
    ftcqt_spectrogram = np.fft.fft(cqt_spectrogram, 2 * number_frequencies - 1, axis=0)
    absftcqt_spectrogram = abs(ftcqt_spectrogram)

    # Derive the CQT envelope and pitch
    cqt_envelope = np.real(
        np.fft.ifft(absftcqt_spectrogram, axis=0)[0:number_frequencies, :]
    )
    cqt_pitch = np.real(
        np.fft.ifft(ftcqt_spectrogram / (absftcqt_spectrogram + 1e-16), axis=0)[
            0:number_frequencies, :
        ]
    )

    return cqt_envelope, cqt_pitch


def cqtsec(
    audio_signal,
    sampling_frequency,
    step_length,
    minimum_frequency=32.70,
    octave_resolution=12,
    number_coefficients=20,
):
    """
    Compute the constant-Q transform (CQT) spectral envelope coefficients (CQT-SEC).

    Inputs:
        audio_signal: audio signal (number_samples,)
        sampling_frequency: sampling frequency in Hz
        step_length: step length in samples
        minimum_frequency: minimum frequency in Hz (default: 32.70 Hz = C1)
        octave_resolution: number of frequency channels per octave (default: 12 frequency channels per octave)
        number_coefficients: number of CQT-SECs (default: 20 coefficients)
    Output:
        cqt_sec: CQT-SECs (number_coefficients, number_frames)
    """

    # Compute the power CQT spectrogram
    cqt_spectrogram = np.power(
        cqt(
            audio_signal,
            sampling_frequency,
            step_length,
            minimum_frequency,
            octave_resolution=12,
        ),
        2,
    )

    # Derive the CQT envelope
    number_frequencies = np.shape(cqt_spectrogram)[0]
    cqt_envelope = np.real(
        np.fft.ifft(
            abs(np.fft.fft(cqt_spectrogram, 2 * number_frequencies - 1, axis=0)), axis=0
        )[0:number_frequencies, :]
    )

    # Extract the CQT-SECs
    coefficient_indices = np.round(
        octave_resolution * np.log2(np.arange(1, number_coefficients + 1))
    ).astype(int)
    cqt_sec = cqt_envelope[coefficient_indices, :]

    return cqt_sec
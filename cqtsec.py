"""
This Python module implements the constant-Q transform spectral envelope coefficients (CQT-SEC) and other related functions.

Functions:
    mfcc - Compute the mel-frequency cepstral coefficients (MFCCs) (using librosa).
    cqtspectrogram - Compute the (magnitude) constant-Q transform (CQT) spectrogram (using librosa).
    cqtdeconv - Deconvolve the constant-Q transform (CQT) spectrogram into a pitch-independent spectral envelope and an energy-normalized pitch component.
    cqtsec - Compute the CQT spectral envelope coefficients (CQT-SEC).

Author:
    Zafar Rafii
    zafarrafii@gmail.com
    http://zafarrafii.com
    https://github.com/zafarrafii
    https://www.linkedin.com/in/zafarrafii/
    08/16/21
"""

import numpy as np
import librosa


def mfcc(
    audio_signal, sampling_frequency, window_length, step_length, number_coefficients=20
):
    """
    Compute the mel-frequency cepstral coefficients (MFCCs) (using librosa).

    Inputs:
        audio_signal: audio signal (number_samples,)
        sampling_frequency: sampling frequency in Hz
        window_length: window length in samples
        step_length: step length in samples
        number_coefficients: number of MFCCs (default: 20 coefficients)
    Output:
        audio_mfcc: audio MFCCs (number_coefficients, number_frames)

    Example: Compute the MFCCs from an audio file.
        # Import the modules
        import numpy as np
        import cqtsec
        import librosa
        import librosa.display
        import matplotlib.pyplot as plt

        # Load the audio signal
        file_path = r'bass_acoustic_000-036-075.wav'
        audio_signal, sampling_frequency = librosa.load(file_path, sr=None, mono=True)

        # Define the parameters and compute the MFCCs
        window_length = pow(2, int(np.ceil(np.log2(0.04 * sampling_frequency))))
        step_length = int(window_length / 2)
        number_coefficients = 20
        audio_mfcc = cqtsec.mfcc(audio_signal, sampling_frequency, window_length, step_length, number_coefficients)

        # Display the MFCCs
        plt.figure(figsize=(14, 4))
        librosa.display.specshow(audio_mfcc, x_axis='time', sr=sampling_frequency, hop_length=step_length, cmap='jet')
        plt.title('MFCCs')
        plt.ylabel('Coefficient')
        plt.tight_layout()
        plt.show()
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


def cqtspectrogram(
    audio_signal,
    sampling_frequency,
    step_length,
    minimum_frequency=32.70,
    octave_resolution=12,
):
    """
    Compute the (magnitude) constant-Q transform (CQT) spectrogram (using librosa).

    Inputs:
        audio_signal: audio signal (number_samples,)
        sampling_frequency: sampling frequency in Hz
        step_length: step length in samples
        minimum_frequency: minimum frequency in Hz (default: 32.70 Hz = C1)
        octave_resolution: number of frequency channels per octave (default: 12 frequency channels per octave)
    Output:
        cqt_spectrogram: magnitude CQT spectrogram (number_frequencies, number_frames)

    Example: Compute the CQT spectrogram from an audio file.
        # Import the modules
        import numpy as np
        import cqtsec
        import librosa
        import librosa.display
        import matplotlib.pyplot as plt

        # Load the audio signal
        file_path = r'bass_acoustic_000-036-075.wav'
        audio_signal, sampling_frequency = librosa.load(file_path, sr=None, mono=True)

        # Define the parameters and compute the CQT spectrogram
        step_length = int(pow(2, int(np.ceil(np.log2(0.04 * sampling_frequency)))) / 2)
        minimum_frequency = 32.70
        octave_resolution = 12
        cqt_spectrogram = cqtsec.cqtspectrogram(audio_signal, sampling_frequency, step_length, minimum_frequency, \
                                                octave_resolution)

        # Display the CQT spectrogram
        plt.figure(figsize=(14, 4))
        librosa.display.specshow(librosa.amplitude_to_db(cqt_spectrogram), x_axis='time', y_axis='cqt_note', \
                                sr=sampling_frequency, hop_length=step_length, fmin=minimum_frequency, \
                                bins_per_octave=octave_resolution, cmap='jet')
        plt.title('CQT spectrogram')
        plt.tight_layout()
        plt.show()
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
    Deconvolve the constant-Q transform (CQT) spectrogram into a pitch-independent spectral envelope and an energy-normalized pitch component.

    Inputs:
        cqt_spectrogram: CQT spectrogram (number_frequencies, number_frames)
    Output:
        cqt_envelope: pitch-independent spectral envelope (number_frequencies, number_frames)
        cqt_pitch: energy-normalized pitch component (number_frequencies, number_frames)

    Example: Deconvolve a CQT spectrogram into its spectral envelope and pitch component.
        # Import the modules
        import numpy as np
        import cqtsec
        import librosa
        import librosa.display
        import matplotlib.pyplot as plt

        # Load the audio signal
        file_path = r'bass_acoustic_000-036-075.wav'
        audio_signal, sampling_frequency = librosa.load(file_path, sr=None, mono=True)

        # Define the parameters and compute the CQT spectrogram
        step_length = int(pow(2, int(np.ceil(np.log2(0.04 * sampling_frequency)))) / 2)
        minimum_frequency = 32.70
        octave_resolution = 12
        cqt_spectrogram = cqtsec.cqtspectrogram(audio_signal, sampling_frequency, step_length, minimum_frequency, \
                                                octave_resolution)

        # Deconvolve the CQT spectrogram into a CQT envelope and pitch
        cqt_envelope, cqt_pitch = cqtsec.cqtdeconv(cqt_spectrogram)

        # Display the CQT spectrogram, envelope, and pitch
        plt.figure(figsize=(14, 4))
        plt.subplot(1, 3, 1)
        librosa.display.specshow(librosa.amplitude_to_db(cqt_spectrogram), x_axis='time', y_axis='cqt_note', \
                                sr=sampling_frequency, hop_length=step_length, fmin=minimum_frequency, \
                                bins_per_octave=octave_resolution, cmap='jet')
        plt.title('CQT spectrogram')
        plt.subplot(1, 3, 2)
        librosa.display.specshow(librosa.amplitude_to_db(cqt_envelope), x_axis='time', y_axis='cqt_note', sr=sampling_frequency, 
                                hop_length=step_length, fmin=minimum_frequency, bins_per_octave=octave_resolution, cmap='jet')
        plt.title('CQT envelope')
        plt.subplot(1, 3, 3)
        librosa.display.specshow(cqt_pitch, x_axis='time', y_axis='cqt_note', sr=sampling_frequency, hop_length=step_length, \
                                fmin=minimum_frequency, bins_per_octave=octave_resolution, cmap='jet')
        plt.title('CQT pitch')
        plt.tight_layout()
        plt.show()
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
        cqtspectrogram(
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
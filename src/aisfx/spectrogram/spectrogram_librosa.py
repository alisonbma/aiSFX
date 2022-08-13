import numpy as np
import librosa

def compute_spectrogram(filename, sr,
                        block_size, hop_size,
                        n_mels,
                        f_min, f_max):
    """Computed Mel spectrograms.
    
    Implemented for convenience in using this package.
    
    May yield lower results compared to original ISMIR code, which uses Essentia.
    
    Args:
        filename (str): Path to audiofile.
        sr (int or float): Sampling rate in Hz.
        block_length (int): Block length in samples.
        hop_length (int): Hop length in samples.
        n_mels (int): Number of mel bins.
        f_min (int or float): Low frequency bound.
        f_max (int or float): High frequency bound.

    Returns:
        np.array: The computed spectrogram.

    """
    db_amin=1e-10
    db_ref=1.0
    dynamic_range=80.0

    y, sr = librosa.load(path=filename, sr=sr)

    # Normalize Audio - Note differences from Essentia, which uses dB: -6 as reference...
    fNorm = np.max(np.abs(y))
    if fNorm == 0:
        fNorm = 1
    y = y / fNorm

    spec = librosa.feature.melspectrogram(y=y,
                                        sr=sr,
                                        n_mels=n_mels,
                                        norm=1.0,
                                        fmin=f_min,
                                        fmax=f_max,
                                        htk=True,
                                        power=2.0,
                                        hop_length=hop_size,
                                        win_length=block_size)
    spec = np.log10(spec * 10000.0 + 1)
    return spec
    
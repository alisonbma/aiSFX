"""
Modified spectrogram computation code from an Essentia library tutorial.
Last accessed February 1, 2022.
"""

# Copyright (C) 2006-2021  Music Technology Group - Universitat Pompeu Fabra
#
# This file is part of Essentia
#
# Essentia is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation (FSF), either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the Affero GNU General Public License
# version 3 along with this program. If not, see http://www.gnu.org/licenses/

import numpy as np
from essentia import Pool
from essentia import run
from essentia.streaming import (EasyLoader, FrameCutter, Windowing, Spectrum,
                                MelBands, UnaryOperator)

def compute_spectrogram(filename, verbose=False, sr=None, block_size=None,
    hop_size=None, window_type='hann', zero_padding=0, f_min=None,
    f_max=None, n_mels=None, warping_formula='htkMel',
    weighting='warping', normalize='unit_sum', bands_type='power', compression_type='shift_scale_log'):
    """Computes the mel spectrogram given the audio filename.
    When the parameter `npy_file` is specified, the data is saved to disk as a numpy array (.npy).
    Use the parameter `force` to overwrite the numpy array in case it already exists.
    The rest of parameters are directly mapped to Essentia algorithms as explained below.
    Note: this functionality is also available as a command line script.
    Parameters:
        filename:
        string - name of audio-file to process
        sr:
        real ∈ (0,inf) (default = 44100)
        the desired output sampling rate [Hz]
        block_size:
        integer ∈ [1,inf) (default = 1024)
        the output frame size
        hop_size:
        integer ∈ [1,inf) (default = 512)
        the hop size between frames
        window_type:
        string ∈ {hamming,hann,hannnsgcq,triangular,square,blackmanharris62,blackmanharris70,blackmanharris74,blackmanharris92} (default = "hann")
        the window type, which can be 'hamming', 'hann', 'triangular', 'square' or 'blackmanharrisXX'
        zero_padding:
        integer ∈ [0,inf) (default = 0)
        the size of the zero-padding
        f_min:
        real ∈ [0,inf) (default = 0)
        a lower-bound limit for the frequencies to be included in the bands
        f_max:
        real ∈ [0,inf) (default = 22050)
        an upper-bound limit for the frequencies to be included in the bands
        n_mels:
        integer ∈ (1,inf) (default = 24)
        the number of output bands
        warping_formula:
        string ∈ {slaneyMel,htkMel} (default = "htkMel")
        The scale implementation type: 'htkMel' scale from the HTK toolkit [2, 3]
        (default) or 'slaneyMel' scale from the Auditory toolbox [4]
        weighting:
        string ∈ {warping,linear} (default = "warping")
        type of weighting function for determining triangle area
        normalize:
        string ∈ {unit_sum,unit_tri,unit_max} (default = "unit_sum")
        spectrum bin weights to use for each mel band: 'unit_max' to make each mel
        band vertex equal to 1, 'unit_sum' to make each mel band area equal to 1
        summing the actual weights of spectrum bins, 'unit_area' to make each
        triangle mel band area equal to 1 normalizing the weights of each triangle
        by its bandwidth
        bands_type:
        string ∈ {magnitude,power} (default = "power")
        'power' to output squared units, 'magnitude' to keep it as the input
        compression_type:
        string ∈ {dB,shift_scale_log,none} (default = "shift_scale_log")
        the compression type to use.
        'shift_scale_log' is log10(10000 * x + 1)
        'dB' is 10 * log10(x)
    Returns:
        (2D array): The mel-spectrogram.
    """

    padded_size = block_size + zero_padding
    spectrum_size = (padded_size) // 2 + 1

    pool = Pool()

    loader = EasyLoader(filename=filename,
                        sampleRate=sr)
    frameCutter = FrameCutter(frameSize=block_size,
                            hopSize=hop_size)
    w = Windowing(zeroPadding=zero_padding,
                type=window_type,
                normalized=False)  # None of the mel bands extraction methods
                                    # we have seen requires window-level normalization.
    spec = Spectrum(size=padded_size)
    mels = MelBands(inputSize=spectrum_size,
                    numberBands=n_mels,
                    sampleRate=sr,
                    lowFrequencyBound=f_min,
                    highFrequencyBound=f_max,
                    warpingFormula=warping_formula,
                    weighting=weighting,
                    normalize=normalize,
                    type=bands_type,
                    log=False)  # Do not compute any compression here.
                                # Use the `UnaryOperator`s methods before
                                # in case a new compression type is required.

    if compression_type.lower() == 'db':
        shift = UnaryOperator(type='identity')
        compressor = UnaryOperator(type='lin2db')

    elif compression_type.lower() == 'shift_scale_log':
        shift = UnaryOperator(type='identity', scale=1e4, shift=1)
        compressor = UnaryOperator(type='log10')

    elif compression_type.lower() == 'none':
        shift = UnaryOperator(type='identity')
        compressor = UnaryOperator(type='identity')

    loader.audio >> frameCutter.signal
    frameCutter.frame >> w.frame >> spec.frame
    spec.spectrum >> mels.spectrum
    mels.bands >> shift.array >> compressor.array >> (pool, 'mel_bands')

    run(loader)
    
    return pool['mel_bands']
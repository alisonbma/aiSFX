import os
import torch

def cuda_select(use_cuda=True):
    """Return `torch.device` using cpu or CUDA.

    Device chosen based on (i) user-desired device and (ii) CUDA availability.

    Args:
        use_cuda (bool): User-desired device. By default, CUDA is used if available.

    Returns:
        str: 'cpu' or 'cuda'.

    """
    if use_cuda == True and torch.cuda.is_available():
        cuda = 'cuda'
    else:
        cuda = 'cpu'
    print('CPU or CUDA: ', cuda)
    return cuda

def get_audioPaths(directory):
    """Return a list of audiofiles to process.

    Recursively collect all paths in directory to parse.
    
    Directory must only contain audiofiles.

    Args:
        directory (str): Directory to folder containing audiofiles to process.

    Returns:
        list[str]: The list of audiofiles to process.

    """
    process_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if not file.startswith('.'):
                process_list.append(os.path.join(root, file))
    assert(len(process_list) > 0)
    return process_list

def spectrogram_selector(spectrogram_type):
    """Selector a spectrogram computation method.
    
    Choices are `Essentia` or `Librosa`.
        
    Args:
        spectrogram_type (str): Compute spectrograms with `'essentia'` or `'librosa'` library.

    Returns:
        fn: Function to compute the spectrograms.
        
    """
    if spectrogram_type == 'essentia':
        from .spectrogram import spectrogram_essentia
        fn = spectrogram_essentia.compute_spectrogram
    else:
        from .spectrogram import spectrogram_librosa
        fn = spectrogram_librosa.compute_spectrogram
    return fn
import numpy as np

def compute_hopSize(block_size, hop_multiplier, data):
    """Compute hop size as a portion of `block_size`.
    
    Use to determine the amount of overlap you want between frames.
        
    Args:
        block_size (int): Block size.
        hop_multiplier (float): The amount of block size to be used as hop size.
        data (np.ndarray): Data that is being blocked.

    Returns:
        int: The computed hop size.
        
    """
    hop_size = block_size * hop_multiplier
    hop_size = int(np.round(hop_size))

    assert(hop_size < data.shape[0])
    assert(hop_size >= 1)
    
    return hop_size

def spectrogram_normalize(spec,
                          spec_norm='original',
                          mn=None, mx=None):
    """Normalize spectrograms.
    
    Uses MinMax normalization.
        
    Args:
        spec (np.ndarray): Spectrogram data.
        spec_norm (str): **'original' uses same cross-dataset training values as in ISMIR paper**, 'local' uses min/max of spec, 'user' allows one to set their own mn and mx value using the respective  arguments.
        mn (float): Minimum spectrogram value for normalization. Will be ignored unless spec_norm='user'.
        mx (float): Maximum spectrogram value for normalization. Will be ignored unless spec_norm='user'.

    Returns:
        np.ndarray: The normalized spectrogram.

    """
    if spec_norm == 'original':
        # Use default from ISMIR paper: Values from all Essentia spectrograms for all datasets in paper
        mn, mx = 0, 9.457798957824707
    elif spec_norm == 'local':
        mn, mx = spec.min(), spec.max() # e.g. 0, 10.0
    elif spec_norm == 'user':
        assert(mn!=None)
        assert(mx!=None)
    else:
        raise('Please enter a supported normalization mode for spectrograms.')    

    assert((mx-mn) > 0)

    return (spec-mn)/(mx-mn)

def blocking(data, block_size, hop_size, drop_partial_block):
    """Block spectrograms.
    
    Chunk the 2D data based on `block_size` and `hop_size`.
    
    Args:
        data (np.array): 2D numpy array to process.
        block_size (int): Number of frames.
        hop_size (int): Number of frames.
        drop_partial_block (bool): Whether to drop the last block if incomplete.

    Returns:
        np.ndarray: The chunked data, np.ndarray([num_blocks, block_size, data.shape[1]]).

    """
    num_blocks = np.ceil((data.shape[0] - block_size) / hop_size + 1).astype(int)
    if (data.shape[0] - block_size) % hop_size != 0 and drop_partial_block == True:
        num_blocks -= 1
    if num_blocks < 1:
        num_blocks = 1

    # pad regardless if necessary...
    pad = (hop_size*(num_blocks-1)+block_size) - data.shape[0]
    data = np.vstack((data, np.zeros((block_size, data.shape[-1]))))

    # split data into blocks
    blocks = [np.array(data[i*hop_size:i*hop_size+block_size]) for i in range(num_blocks)]

    return np.stack(blocks)

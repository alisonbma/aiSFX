from aisfx.preprocessing import *
from aisfx.utils import *
from aisfx.models.cnn2d import EmbeddingNet, CrossNet
import os
import argparse
from tqdm import tqdm
import numpy as np
import torch

# Sample rate for preprocessing
SAMPLE_RATE = 44100.0

# Input Spectrogram Computation
SPEC_BLOCK_LENGTH, SPEC_HOP_LENGTH = 2048, 1024
MEL_BINS = 96
F_MIN, F_MAX = 0.0, 22050.0

# Embeddings block length
# Approximately 2 seconds of audio: 1 / SAMPLE_RATE * HOP_SIZE = 23ms
EMB_BLOCK_LENGTH = 100

def get_path_modelWeights():
    """Return relative path to ``aiSFX`` package with best model weights.

    Path to .pickle file encompassing the best ray.tune checkpoint of model **X-Sequential-CE**.

    Returns:
        str: Path to weights.

    """
    return os.path.join(os.path.dirname(__file__), 'models/checkpoint.pth')

def load_bestModel(weights, ds_dict, cuda):
    """Return best model: **X-Sequential-CE**.

    Re-loads best CrossNet model, **X-Sequential-CE** with saved weights on specified `torch.device()`.

    Args:
        weights (torch.model.state_dict): PyTorch model weights.
        ds_dict (dict): Dictionary of dataset names and number of classes for all datasets used in cross-dataset training.
        cuda (str): 'cpu' or 'cuda'.

    Returns:
        nn.Module: PyTorch model.

    """
    model = CrossNet(EmbeddingNet('max', 3, 1, 64), ds_dict, cuda)
    model.load_state_dict(weights)
    model.to(cuda)
    model.eval()
    return model

def load_weights(cuda):
    """Load PyTorch model weights.

    Loads weights using the path to a saved model checkpoint.

    Args:
        cuda (str): 'cpu' or 'cuda'.

    Returns:
        torch.model.state_dict: PyTorch model weights.
        dict: Dictionary of dataset names and number of classes for all datasets used in cross-dataset training.

    """
    path_checkpoint = get_path_modelWeights()
    if cuda == 'cuda':
        weights, ds_dict = torch.load(path_checkpoint)
    else:
        weights, ds_dict = torch.load(path_checkpoint, map_location=torch.device('cpu'))
    return weights, ds_dict


def model_get_embedding(spec,
                        emb_hop_size,
                        drop_partial_block,
                        model,
                        cuda):
    """Extract embeddings using a PyTorch model.
    
    Processes Tensor spectrogram input through PyTorch model on specified `torch.device()`.
        
    Args:
        spec (np.ndarray): 2D Spectrogram data.
        emb_hop_size (float): Embedding hop size, a multiplier of the embedding block size.
        drop_partial_block (bool): Whether to drop the last block if incomplete.
        model (torch.nn.Module): PyTorch model.
        cuda (str): 'cpu' or 'cuda'.

    Returns:
        np.ndarray: The extracted embeddings, np.ndarray([num_frames, 512]), where num_frames is the number of frames based on EMB_BLOCK_LENGTH and emb_hop_size, and 512 is the embedding dimensionality.
        
    """
    # Block the spectrograms according to the amount of duration you wish to input into the model.
    blocks = blocking(spec,
                      block_size=EMB_BLOCK_LENGTH,
                      hop_size=compute_hopSize(EMB_BLOCK_LENGTH, emb_hop_size, spec),
                      drop_partial_block=drop_partial_block)

    with torch.no_grad():
        if type(blocks) != type(torch.Tensor()):
            data = torch.Tensor(blocks).to(cuda)
        output = model.get_embedding(data).cpu()

    return output

def main(dir_audio, dir_export,
         spectrogram_type='essentia',
         spec_norm='original', norm_mn = None, norm_mx = None,
         drop_partial_block=True,
         emb_hop_size=0.5,
         use_cuda=True):
    """Extract and save embeddings from all files in a directory.
    
    Uses the **X-Sequential-CE** model from **ISMIR** paper.

    Args:
        dir_audio (str): Path to audio directory for processing.
        dir_export (str): Path to directory for saving extracted embeddings.
        spectrogram_type (str): Compute spectrograms with `'essentia'` or `'librosa'` library. **Original ISMIR paper uses `essentia`**.
        emb_hop_size (float): Embedding hop size, a multiplier of the embedding block size. **Original ISMIR paper uses 0.5 (1s hop)**.
        use_cuda (str): 'cpu' or 'cuda'.
        
    """

    # Parse directory for files to process (all files in directory must be audio-files)
    process_list = get_audioPaths(dir_audio)

    # Choose between Essentia or Librosa for extracting spectrograms
    spectrogram_compute = spectrogram_selector(spectrogram_type)

    # Load best model and weights
    cuda = cuda_select(use_cuda=use_cuda) # CPU or GPU support
    weights, ds_dict = load_weights(cuda) # Get best weights from X-Sequential-CE
    model = load_bestModel(weights, ds_dict, cuda) # Load best weights onto CrossNet (cross-dataset training model architecture)

    # Create directory to store extracted embeddings
    os.makedirs(dir_export, exist_ok=True)

    # Process each audio-file one by one...
    for audio in tqdm(process_list):       
        
        # Compute spectrograms
        spec = spectrogram_compute(filename=audio,
                                   sr=SAMPLE_RATE,
                                   block_size=SPEC_BLOCK_LENGTH,
                                   hop_size=SPEC_HOP_LENGTH,
                                   n_mels=MEL_BINS,
                                   f_min=F_MIN,
                                   f_max=F_MAX)

        # Normalize spectrograms
        spec = spectrogram_normalize(spec,
                                     spec_norm=spec_norm,
                                     mn=norm_mn,
                                     mx=norm_mx) # currently using default essentia normalization

        # Extract embeddings with model
        embedding = model_get_embedding(spec,
                                        emb_hop_size,
                                        drop_partial_block,
                                        model,
                                        cuda)
        
        # Save embeddings as numpy file
        filename_savedEmbedding = os.path.splitext(os.path.split(audio)[-1])[0] + '.npy'
        path_savedEmbedding = os.path.join(dir_export, filename_savedEmbedding)
        np.save(path_savedEmbedding, embedding)

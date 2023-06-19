# Install 

## Dependencies

1. Download [Anaconda](https://www.anaconda.com/)
2. Create a conda environment
```
# models were trained using python==3.9.6
conda create -n myCondaEnvironment python=3.9.6
```
3. Install [PyTorch](https://pytorch.org/get-started/locally/) with or without CUDA support. This library was developed with `torch==1.12.0`.
4. Download the aiSFX package with either method
```
# Using the released pip package
pip install aisfx
```
```
# Using the latest updated code
pip install git+https://github.com/alisonbma/aiSFX.git
```

5. To replicate the ISMIR paper code, you will need to download the [Essentia](https://essentia.upf.edu/) library to compute the spectrograms.

```

pip install essentia==2.1b6.dev778

```

## Operating System Compatibility
```{warning}
- As of the moment, Essentia does support installation on MacOS. However, installation is only supported on Homebrew, which does not work well with conda environments. Thus, to re-create our results, we suggest you use Linux or WSL on Windows. If you still wish to do so, refer to the API Reference and use {py:func}`aisfx.inference.main` as a reference for any modifications to the embedding extraction code.
- For convenience, we have also implemented an option to run the code with [Librosa](https://librosa.org/), see {py:func}`aisfx.spectrogram.spectrogram_librosa.compute_spectrogram`. This may produce lower results.
```

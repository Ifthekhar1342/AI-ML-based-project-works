# Generative AI for 3D CAD Model Reconstruction and Latent-Space Morphing

## Objective
To reconstruct 3D CAD models from input data and interpolate between different 3D CAD designs in a meaningful way.

## Dataset
<p align="justify">
The dataset of this work is taken from the <a href="https://modelnet.cs.princeton.edu/">ModelNet10</a>, which contains 10 categories of CAD models (i.e. chair, bed) along with .off mesh files.
</p>

## Project workflow
The code of this project [![Open In Colab](https://colab.research.google.com/drive/1GciPX32je-ByT1K6bG9nM36BGGeoGIg3)](https://colab.research.google.com/drive/1GciPX32je-ByT1K6bG9nM36BGGeoGIg3)
<p align="justify">
The implementation is utilized to build a generative model, which is capable of learning latent space of 3D CAD models. This process includes:

### Step 01: Data Preprocessing
The 3D meshes (in .off format) are converted into voxel grids by allowing the neural network to process spatial data. The "Encoder" compresses the 3D input into a low-dimensional latent vector and a "Decoder" attempts to reconstruct the original 3D shape from this vector.

### Step 02: Reconstruction
A Variational Autoencoder (VAE) model is trained to minimize the difference between the input object and it's reconstructed part.

### Step 03: Morphing
The decoder generates 3D CAD shapes by interpolating between two points in the latent space through a visual transition.


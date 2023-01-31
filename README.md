# Dimensionality Reduction in Infrared Microscopy
[![GitHub stars](https://img.shields.io/github/stars/RUB-Bioinf/DimensionalityReduction.svg?style=social&label=Star)](https://github.com/RUB-Bioinf/DimensionalityReduction)
[![GitHub forks](https://img.shields.io/github/forks/RUB-Bioinf/DimensionalityReduction.svg?style=social&label=Fork)](https://github.com/RUB-Bioinf/DimensionalityReduction)
[![GitHub Downloads](https://img.shields.io/github/downloads/RUB-Bioinf/DimensionalityReduction/total?style=social)](https://github.com/RUB-Bioinf/DimensionalityReduction/releases) 
&nbsp; 
[![Follow us on Twitter](https://img.shields.io/twitter/follow/MuellerDajana?style=social&logo=twitter)](https://twitter.com/intent/follow?screen_name=MuellerDajana)


Dimensionality reduction in infrared microscopy is aiming to preserve both spatial and molecular information in a more compact manner and therefore reduces the computational time for subsequent classification or segmentation tasks. We offer four different dimensionality reduction approaches mentioned in *Dimensionality reduction for deep learning in infrared microscopy: A comparative computational survey*, namely Principle Component Analysis, Uniform Manifold Approximation and Projection and two different Contractive Autoencoder.

![Workflow_DimRed](/img/Workflow_DimRed.png?raw=true "Approach Overview")

***

[![Generic badge](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](docs/contribute.md)
![Size](https://img.shields.io/github/repo-size/RUB-Bioinf/DimensionalityReduction?style=plastic)
[![Language](https://img.shields.io/github/languages/top/RUB-Bioinf/DimensionalityReduction?style=plastic)](https://github.com/RUB-Bioinf/DimensionalityReduction)

***




## Overview

This repository contains four different scripts:
- PCA_tf2.py  : Performs principle component analysis on spectral data of shape (X\*Y,s pectra) with *n_components*. Data can be standarized such that they have a mean of 0 and standard deviation of 1. 
- UMAP_tf2.py : Performs uniform manifold approximation and projection on data of shape (X\*Y, spectra) with *n_neighbors, n_components, min_dist and metric*. Data can be standarized such that they have a mean of 0 and standard deviation of 1. 
- FCCAE_tf2.py: Training of the entire stacked autoencoder from scratch, yielding a fully connected contractive autoencoder with Tensorflow.
- SCAE_tf2.py : Training of a series of stacked contractive autoencoders which are trained with one hidden layer each and are afterwards connected to form a deep autoencoder with Tensorflow.


***



## Usage
Learn more [here](https://github.com/RUB-Bioinf/DimensionalityReduction/wiki/Usage).



## Correspondence

[**Prof. Dr. Axel Mosig**](mailto:axel.mosig@rub.de): Bioinformatics, Center for Protein Diagnostics (ProDi), Ruhr-University Bochum, Bochum, Germany

http://www.bioinf.rub.de/

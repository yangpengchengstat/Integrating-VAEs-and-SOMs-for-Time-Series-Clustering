# Integrating VAEs and SOMs for Time Series Clustering

This repository accompanies the chapter **“Integrating VAEs and SOMs for Time Series Clustering,”** which explores a novel framework (DVESOM) that combines:
1. **Long Short-Term Memory (LSTM)** networks for temporal modeling,
2. **Variational Autoencoders (VAEs)** for generative representation learning, and
3. **Self-Organizing Maps (SOMs)** for topologically preserving cluster structures.

---

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [References](#references)
- [License](#license)

---

## Overview

**Deep Variational Embedded Self-Organizing Map (DVESOM)** is designed to tackle the challenges inherent in clustering long-sequence time-series data. By leveraging LSTM layers, the model captures both short-term and long-term dependencies in sequences; by integrating a VAE, the framework learns a robust, generative latent representation; and by embedding a SOM, it preserves topological relationships in the latent space, leading to interpretable and coherent clusters.

**Key advantages** of the approach:
- **Temporal Encoding**: LSTM-based encoder handles complex, nonstationary time-series patterns.
- **Generative Latent Space**: VAE ensures a flexible, probabilistic representation suitable for reconstruction and sampling.
- **Topological Clustering**: SOM constraints impose a neighborhood-preserving structure, improving interpretability and cluster coherence.

---

## Key Features

1. **Bidirectional LSTM Encoder**  
   Captures forward and backward context to enhance modeling of time-series data.

2. **VAE with Reparameterization Trick**  
   Learns a probabilistic latent embedding, balancing reconstruction fidelity and prior regularization (KL divergence).

3. **SOM Backpropagation**  
   Uses a differentiable SOM distortion term, allowing both the codebook vectors and the encoder parameters to be updated jointly via gradient-based methods.

4. **End-to-End Training**  
   Merges the entire pipeline (LSTM, VAE, and SOM) into a unified loss function, ensuring synergy among temporal encoding, generative modeling, and topological constraints.

---

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/YourUsername/TimeSeries-VAE-SOM.git
   cd TimeSeries-VAE-SOM

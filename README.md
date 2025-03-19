# Spectral Kurtosis for RFI Detection

## Overview

This repository contains code for implementing Frequency Separated Spectral Kurtosis (FSSK), a novel method for calculating spectral kurtosis without the need for Fast Fourier Transform (FFT). The technique, developed as part of our research, enhances real-time detection of Radio Frequency Interference (RFI) using Software Defined Radio (SDR) technology.

The method and its applications are described in our paper published in *Proceedings of Science 2024*:

> **An Evaluation of a New Method of Calculating RFI with Kurtosis**  
> *Sylvia Llosa, Arvind Aradhya, Kevin Gifford*

## Features

- **Real-time RFI detection** with increased computational efficiency  
- **Bypasses FFT**, leveraging SDR for direct frequency separation  
- **Applicable to radio astronomy and communication systems**  
- **Optimized for Raspberry Pi and other embedded platforms**  

## Installation

### Dependencies

Ensure you have the following installed:

- Python 3.x  
- NumPy  
- SciPy  
- Matplotlib (for visualization)  
- RTL-SDR drivers (if using an SDR for data acquisition)  

### Setup

```sh
# Clone the repository
git clone https://github.com/your-repo/spectral-kurtosis.git
cd spectral-kurtosis

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running the Kurtosis Calculation

```sh
python run_kurtosis.py --input data/sample_iq_data.npy --output results/kurtosis_output.png
```

### Real-time Processing with SDR

If using an SDR device:

```sh
python run_sdr_kurtosis.py --device 0 --freq 915e6 --rate 2e6 --gain 40
```

## Methodology

Traditional spectral kurtosis calculations rely on FFT to convert IQ data into the frequency domain before analysis. Our method instead applies kurtosis analysis directly on separated frequency components obtained from an SDR, significantly reducing computational overhead. 

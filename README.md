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

MATLAB (or run in MATLAB browser)

### Setup

```sh
# Clone the repository
git clone https://github.com/your-repo/spectral-kurtosis.git
cd spectral-kurtosis
```

## Usage

**Kurtosis.m**  : main file for simulated experiements  
**FiveMethods/SpectrualKurtosisMethods.m**: Test code for the five methods of spectral kurtosis  
**ROC_curves**: Test code for creating Receiver Operating Characteristic (ROC) plots  
**spectral_kurtosis_example.m**: Test code for creating signal environment and using builtin spectral kurtosis functions  
**kurtosisHistograms.m**: Used to generate histograms for paper  
**SpeedAccuracyVariability.m**: used to measure the exections times of each method  

### Running the Kurtosis Calculation
matlab console enter 'matlab_filename'
```sh
Kurtosis
```


## Methodology

Traditional spectral kurtosis calculations rely on FFT to convert IQ data into the frequency domain before analysis. Our method instead applies kurtosis analysis directly on separated frequency components obtained from an SDR, significantly reducing computational overhead. 

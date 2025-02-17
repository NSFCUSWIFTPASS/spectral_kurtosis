% 1. Periodogram-based spectral kurtosis (PK)
%   - Calculate the periodogram of the time series data, normalize it, and estimate the fourth moment of the normalized periodogram as described in my previous answer.
%   - PK is sensitive to non-stationarity and leakage effects, but it's computationally efficient.
%
% 2. Periodogram of squares-based spectral kurtosis (PKS)
%   - First, square the time series data and calculate the periodogram of squared values.
%   - Normalize the periodogram, estimate the fourth moment, and divide by the number of frequency bins as in PK.
%   - PKS is less sensitive to non-stationarity than PK, but it's more
%   computationally intensive.
%
% 3. Modified periodogram of squares-based spectral kurtosis (MPKS)
%   - Similar to PKS, but use the modified periodogram instead of the regular one. The modified periodogram is calculated by first estimating the autocorrelation function and
%then taking its square root.
%   - MPKS reduces the impact of leakage effects compared to PK and PKS, but it's even more computationally intensive.
%
% 4. Cross-spectral kurtosis (CSK)
%   - Calculate the cross-periodogram between the original time series data and its lagged version, normalize the result, and estimate the fourth moment of the normalized
%cross-periodogram.
%   - CSK can account for non-stationarity in the data, but it requires more computational resources than PK, PKS, or MPKS.
%
% 5. Modified cross-spectral kurtosis (MCSK)
%   - Similar to CSK, but use the modified periodogram instead of the regular one.
%   - MCSK reduces the impact of leakage effects compared to CSK and is less sensitive to non-stationarity, but it's even more computationally intensive than CSK.

% Clear workspace and command window
clear; clc; close all;

% Sampling parameters
fs = 1000; % Sampling frequency in Hz
T = 10; % Duration in seconds
t = (0:1/fs:T)'; % Time vector (column vector)

% Chirp signal parameters
f_min = 50; % Start frequency in Hz
f_max = 600; % End frequency in Hz

% Generate a chirp signal with white Gaussian noise
x = chirp(t, f_min, T, f_max, 'linear') + randn(size(t));

% Define FFT size
NFFT = 256;

% 1. Periodogram-based Spectral Kurtosis (PK)
[Pxx, F] = pwelch(x, hamming(NFFT), [], NFFT, fs);
sk_pk = (Pxx.^4) ./ (mean(Pxx).^2);

% 2. Periodogram of Squares-based Spectral Kurtosis (PKS)
x_squared = x.^2;
[Pxx_squared, F_squared] = pwelch(x_squared, hamming(NFFT), [], NFFT, fs);
sk_pks = (Pxx_squared.^4) ./ (mean(Pxx_squared).^2);

% 3. Modified Periodogram of Squares-based Spectral Kurtosis (MPKS)
acf_x = xcov(x_squared, 'biased'); 
sqrt_acf_x = sqrt(abs(acf_x));
[Pxx_mpks, F_mpks] = pwelch(sqrt_acf_x, hamming(NFFT), [], NFFT, fs);
sk_mpks = (Pxx_mpks.^4) ./ (mean(Pxx_mpks).^2);

% 4. Cross-Spectral Kurtosis (CSK)
y = chirp(t, f_min, T, f_max, 'linear') + randn(size(t)); 
[Pxy, Fxy] = cpsd(x, y, hamming(NFFT), [], NFFT, fs);
sk_csk = (abs(Pxy).^4) ./ (mean(abs(Pxy)).^2);

% 5. Modified Cross-Spectral Kurtosis (MCSK)
acf_yx = xcov(x.^2, y.^2, 'biased'); 
sqrt_acf_yx = sqrt(abs(acf_yx));
[Pxx_mcsk, F_mcsk] = pwelch(sqrt_acf_yx, hamming(NFFT), [], NFFT, fs);
sk_mcsk = (Pxx_mcsk.^4) ./ (mean(Pxx_mcsk).^2);

%% **Single Comparison Subplot**
figure;
subplot(2,3,1);
plot(F, sk_pk, 'b');
title('PK - Periodogram SK');
xlabel('Frequency (Hz)');
ylabel('Kurtosis');
grid on;

subplot(2,3,2);
plot(F_squared, sk_pks, 'r');
title('PKS - Periodogram of Squares SK');
xlabel('Frequency (Hz)');
ylabel('Kurtosis');
grid on;

subplot(2,3,3);
plot(F_mpks, sk_mpks, 'm');
title('MPKS - Modified Periodogram of Squares SK');
xlabel('Frequency (Hz)');
ylabel('Kurtosis');
grid on;

subplot(2,3,4);
plot(Fxy, sk_csk, 'g');
title('CSK - Cross-Spectral SK');
xlabel('Frequency (Hz)');
ylabel('Kurtosis');
grid on;

subplot(2,3,5);
plot(F_mcsk, sk_mcsk, 'k');
title('MCSK - Modified Cross-Spectral SK');
xlabel('Frequency (Hz)');
ylabel('Kurtosis');
grid on;

sgtitle('Comparison of Spectral Kurtosis Methods');

%% **Separate Figures for Each Method**
figure;
plot(F, sk_pk, 'b');
title('Periodogram-based Spectral Kurtosis (PK)');
xlabel('Frequency (Hz)');
ylabel('Kurtosis');
grid on;

figure;
plot(F_squared, sk_pks, 'r');
title('Periodogram of Squares-based Spectral Kurtosis (PKS)');
xlabel('Frequency (Hz)');
ylabel('Kurtosis');
grid on;

figure;
plot(F_mpks, sk_mpks, 'm');
title('Modified Periodogram of Squares-based Spectral Kurtosis (MPKS)');
xlabel('Frequency (Hz)');
ylabel('Kurtosis');
grid on;

figure;
plot(Fxy, sk_csk, 'g');
title('Cross-Spectral Kurtosis (CSK)');
xlabel('Frequency (Hz)');
ylabel('Kurtosis');
grid on;

figure;
plot(F_mcsk, sk_mcsk, 'k');
title('Modified Cross-Spectral Kurtosis (MCSK)');
xlabel('Frequency (Hz)');
ylabel('Kurtosis');
grid on;

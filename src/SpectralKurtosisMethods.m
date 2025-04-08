clc; clear; close all;

% Define parameters
fs = 1000;      % Sampling frequency (Hz)
T = 10;         % Duration (seconds)
t = 0:1/fs:T;   % Time vector

% Chirp parameters
f0 = 0.001;     % Start frequency (Hz)
f1 = 10;        % End frequency (Hz)

% Generate chirp signal with logarithmic frequency sweep
chirp_signal = chirp(t, f0, max(t), f1, 'logarithmic');

% Generate white Gaussian noise
noise = randn(size(t));

% Generate chirp signal with noise (scaled noise level)
SNR = 3; % Signal-to-noise ratio (adjust as needed)
noise_power = var(chirp_signal) / SNR;
scaled_noise = sqrt(noise_power) * noise;
chirp_with_noise = chirp_signal + scaled_noise;

% Compute kurtosis values
kurt_chirp_noise = kurtosis(chirp_with_noise);
kurt_noise = kurtosis(noise);
kurt_chirp = kurtosis(chirp_signal);

% Create figure
figure;

% Plot row 1: Time-domain signals
subplot(2,3,1);
plot(t, chirp_with_noise, 'b');
title('Chirp Signal with White Gaussian Noise');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;
xlim([0 T]);

subplot(2,3,2);
plot(t, noise, 'b');
title('White Gaussian Noise');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;
xlim([0 T]);

subplot(2,3,3);
plot(t, chirp_signal, 'b');
title('Chirp Signal');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;
xlim([0 T]);

% Plot row 2: Histograms with kurtosis values
subplot(2,3,4);
histogram(chirp_with_noise, 50);
title(sprintf('Histogram of Chirp + Noise\nKurtosis: %.4f', kurt_chirp_noise));
xlabel('Amplitude');
ylabel('Frequency');
grid on;

subplot(2,3,5);
histogram(noise, 50);
title(sprintf('Histogram of White Gaussian Noise\nKurtosis: %.4f', kurt_noise));
xlabel('Amplitude');
ylabel('Frequency');
grid on;

subplot(2,3,6);
histogram(chirp_signal, 50);
title(sprintf('Histogram of Chirp Signal\nKurtosis: %.4f', kurt_chirp));
xlabel('Amplitude');
ylabel('Frequency');
grid on;

% Adjust layout
sgtitle('Comparison of Spectral Kurtosis Methods with Three Different Noise Signals');

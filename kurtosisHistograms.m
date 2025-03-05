%% 
clc; clear; close all;

% Generate signals
fs = 1000; % Sampling frequency
t = 0:1/fs:1-1/fs; % Time vector
chirp_signal = chirp(t,10,1,100,'linear');
gaussian_noise = randn(size(t));
chirp_with_noise = chirp_signal + gaussian_noise;
white_gaussian_noise = randn(size(t));

signals = {chirp_with_noise, white_gaussian_noise, chirp_signal};
labels = {'Chirp + Noise', 'White Gaussian Noise', 'Chirp'};

figure;
num_methods = 5; % Number of kurtosis methods
num_cols = 3; % Number of signal types
num_bins = 50; % Bins for histogram

for i = 1:num_cols
    signal = signals{i};
    
    % 1. Traditional Kurtosis
    subplot(num_methods, num_cols, i);
    histogram(signal, num_bins, 'Normalization', 'pdf');
    title(['Traditional Kurtosis: ', labels{i}]);
    
    % 2. Time-Domain Voltage Kurtosis
    voltage_deviation = abs(signal - mean(signal));
    subplot(num_methods, num_cols, num_cols + i);
    histogram(voltage_deviation, num_bins, 'Normalization', 'pdf');
    title(['Voltage Kurtosis: ', labels{i}]);
    
    % 3. Time-Domain Power Kurtosis (Instantaneous Power)
    power_inst = abs(signal).^2;
    subplot(num_methods, num_cols, 2*num_cols + i);
    histogram(power_inst, num_bins, 'Normalization', 'pdf');
    title(['Instantaneous Power Kurtosis: ', labels{i}]);
    
    % 4. Time-Domain Power Kurtosis (Integrated Power)
    window_size = 100;
    power_integrated = movmean(power_inst, window_size);
    subplot(num_methods, num_cols, 3*num_cols + i);
    histogram(power_integrated, num_bins, 'Normalization', 'pdf');
    title(['Integrated Power Kurtosis: ', labels{i}]);
    
    % 5. Frequency Separated Spectral Kurtosis (Simulated Filtering)
    num_bands = 5;
    f0 = 10; f1 = 100;
    band_edges = linspace(f0, f1, num_bands+1);
    kurt_fssk = zeros(1, num_bands);
    
    for j = 1:num_bands
        band_signal = signal .* sin(2 * pi * band_edges(j) * t); % Simulated filtering
        kurt_fssk(j) = kurtosis(band_signal);
    end
    subplot(num_methods, num_cols, 4*num_cols + i);
    histogram(kurt_fssk, num_bins, 'Normalization', 'pdf');
    title(['FSSK Kurtosis: ', labels{i}]);
end

sgtitle('Spectral Kurtosis Analysis for Five Different Methods');

fs = 1000;           % Sampling frequency (Hz)
duration = 10;      % Duration of signal (seconds)
num_samples = fs * duration;

% Generate white noise
white_noise = randn(num_samples, 1);

% Plotting the white noise
figure;
plot(white_noise(1:1000)); % Plot first 1000 samples
title('White Noise Signal');
xlabel('Sample');
ylabel('Amplitude');

% Compute I/Q values (simulate SDR output)
I = real(white_noise);
Q = imag(hilbert(white_noise));
IQ = I + 1j*Q;

% 1. Time-Domain Voltage Kurtosis
voltage_start = tic;
voltage_deviation = abs(IQ - mean(IQ));
kurt_voltage = kurtosis(voltage_deviation);
voltage_time = toc(voltage_start);

% 2. Time-Domain Power Kurtosis (Instantaneous Power)
power_inst_start = tic;
power_inst = abs(IQ).^2;
kurt_power_inst = kurtosis(power_inst);
power_inst_time = toc(power_inst_start);

% 3. Time-Domain Power Kurtosis (Integrated Power)
power_integrated_start = tic;
window_size = 100; % Integration window length
power_integrated = movmean(power_inst, window_size);
kurt_power_integrated = kurtosis(power_integrated);
power_integrated_time = toc(power_integrated_start);

% 4. Traditional Spectral Kurtosis (FFT-based)
sk_start = tic;
[skurt, time_direct] = spectral_kurtosis(white_noise, fs);
sk_time = toc(sk_start);

% Define filter parameters
low_cutoff = 100; % Low cutoff frequency (Hz)
high_cutoff = 300; % High cutoff frequency (Hz)
[b, a] = butter(4, [low_cutoff, high_cutoff] / (fs / 2), 'bandpass');
filtered_noise = filter(b, a, white_noise);

% 5. Frequency Separated Spectral Kurtosis (FSSK)
fssk_start = tic;
[skurt_filtered, time_filtered] = spectral_kurtosis(filtered_noise, fs);
fssk_time = toc(fssk_start);

% Additional Metrics
% Accuracy against theoretical kurtosis (Gaussian noise should be 3)
theo_kurt = 3;
acc_voltage = abs(kurt_voltage - theo_kurt);
acc_power_inst = abs(kurt_power_inst - theo_kurt);
acc_power_integrated = abs(kurt_power_integrated - theo_kurt);
acc_sk = abs(skurt - theo_kurt);
acc_fssk = abs(skurt_filtered - theo_kurt);

% Statistical Variability
num_trials = 100;
var_voltage = var(arrayfun(@(x) kurtosis(abs(randn(num_samples,1)-mean(randn(num_samples,1)))), 1:num_trials));
var_power_inst = var(arrayfun(@(x) kurtosis(abs(randn(num_samples,1)).^2), 1:num_trials));
var_power_integrated = var(arrayfun(@(x) kurtosis(movmean(abs(randn(num_samples,1)).^2, window_size)), 1:num_trials));
var_sk = var(cell2mat(arrayfun(@(x) spectral_kurtosis(randn(num_samples,1), fs), 1:num_trials, 'UniformOutput', false)));
var_fssk = var(cell2mat(arrayfun(@(x) spectral_kurtosis(filter(b,a,randn(num_samples,1)), fs), 1:num_trials, 'UniformOutput', false)));

% Display results in a formatted table
fprintf('\n%-40s %-20s', 'Method', 'Kurtosis Value');
fprintf('------------------------------------------------------------\n');
fprintf('%-40s %-20.5f\n', 'Time-Domain Voltage Kurtosis', kurt_voltage);
fprintf('%-40s %-20.5f\n', 'Instantaneous Power Kurtosis', kurt_power_inst);
fprintf('%-40s %-20.5f\n', 'Integrated Power Kurtosis', kurt_power_integrated);
fprintf('%-40s %-20.5f\n', 'FFT-based Spectral Kurtosis', skurt);
fprintf('%-40s %-20.5f\n', 'Frequency Separated Spectral Kurtosis (FSSK)', skurt_filtered);

fprintf('\n%-40s %-20s', 'Method', 'Computation Time (s)');
fprintf('------------------------------------------------------------\n');
fprintf('%-40s %-20.5f\n', 'Time-Domain Voltage Kurtosis', voltage_time);
fprintf('%-40s %-20.5f\n', 'Instantaneous Power Kurtosis', power_inst_time);
fprintf('%-40s %-20.5f\n', 'Integrated Power Kurtosis', power_integrated_time);
fprintf('%-40s %-20.5f\n', 'FFT-based Spectral Kurtosis', sk_time);
fprintf('%-40s %-20.5f\n', 'Frequency Separated Spectral Kurtosis (FSSK)', fssk_time);

fprintf('\n%-40s %-20s', 'Method', 'Accuracy (Deviation from 3)');
fprintf('------------------------------------------------------------\n');
fprintf('%-40s %-20.5f\n', 'Time-Domain Voltage Kurtosis', acc_voltage);
fprintf('%-40s %-20.5f\n', 'Instantaneous Power Kurtosis', acc_power_inst);
fprintf('%-40s %-20.5f\n', 'Integrated Power Kurtosis', acc_power_integrated);
fprintf('%-40s %-20.5f\n', 'FFT-based Spectral Kurtosis', acc_sk);
fprintf('%-40s %-20.5f\n', 'Frequency Separated Spectral Kurtosis (FSSK)', acc_fssk);

fprintf('\n%-40s %-20s', 'Method', 'Statistical Variability');
fprintf('------------------------------------------------------------\n');
fprintf('%-40s %-20.5f\n', 'Time-Domain Voltage Kurtosis', var_voltage);
fprintf('%-40s %-20.5f\n', 'Instantaneous Power Kurtosis', var_power_inst);
fprintf('%-40s %-20.5f\n', 'Integrated Power Kurtosis', var_power_integrated);
fprintf('%-40s %-20.5f\n', 'FFT-based Spectral Kurtosis', var_sk);
fprintf('%-40s %-20.5f\n', 'Frequency Separated Spectral Kurtosis (FSSK)', var_fssk);

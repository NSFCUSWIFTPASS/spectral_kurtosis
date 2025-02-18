% MATLAB Script for Spectral Kurtosis Calculation using Five Methods

% Parameters for chirp signal
fs = 1e4; % Sampling frequency
T = 1; % Duration (seconds)
t = 0:1/fs:T-1/fs; % Time vector
f0 = 500; % Start frequency of chirp
f1 = 2500; % End frequency of chirp
signal = chirp(t, f0, T, f1, '', 90);

% Compute I/Q values (simulate SDR output)
I = real(signal);
Q = imag(hilbert(signal));
IQ = I + 1j*Q;

% 1. Time-Domain Voltage Kurtosis
voltage_deviation = abs(IQ - mean(IQ));
kurt_voltage = kurtosis(voltage_deviation);

% 2. Time-Domain Power Kurtosis (Instantaneous Power)
power_inst = abs(IQ).^2;
kurt_power_inst = kurtosis(power_inst);

% 3. Time-Domain Power Kurtosis (Integrated Power)
window_size = 100; % Integration window length
power_integrated = movmean(power_inst, window_size);
kurt_power_integrated = kurtosis(power_integrated);

% 4. Traditional Spectral Kurtosis (FFT-based)
NFFT = 1024;
X = fft(signal, NFFT);
X_power = abs(X).^2;
kurt_fft = kurtosis(X_power);

% 5. Frequency Separated Spectral Kurtosis (FSSK, simulated via filtering)
num_bands = 5; % Number of frequency bands
band_edges = linspace(f0, f1, num_bands+1);
kurt_fssk = zeros(1, num_bands);

for i = 1:num_bands
    bandpass_filt = designfilt('bandpassfir', 'FilterOrder', 50, ...
        'CutoffFrequency1', band_edges(i), 'CutoffFrequency2', band_edges(i+1), ...
        'SampleRate', fs);
    filtered_signal = filter(bandpass_filt, signal);
    filtered_power = abs(filtered_signal).^2;
    kurt_fssk(i) = kurtosis(filtered_power);
end

% Normalize values for better visualization
voltage_deviation = voltage_deviation / max(voltage_deviation);
power_inst = power_inst / max(power_inst);
power_integrated = power_integrated / max(power_integrated);
X_power = X_power / max(X_power);

% Plot results with enhanced clarity
figure;
subplot(2,3,1);
plot(t, signal);
title('Quadratic Chirp Signal'); xlabel('Time (s)'); ylabel('Amplitude');

subplot(2,3,2);
xlabel('Method Index');
bar(1, kurt_voltage);
title('Time-Domain Voltage Kurtosis'); xlabel('Time (s)'); ylabel('Kurtosis Value');
xticks([]);

subplot(2,3,3);
xlabel('Method Index');
bar(1, kurt_power_inst);
title('Instantaneous Power Kurtosis'); xlabel('Time (s)'); ylabel('Kurtosis Value');
xticks([]);

subplot(2,3,4);
xlabel('Method Index');
bar(1, kurt_power_integrated);
title('Integrated Power Kurtosis'); xlabel('Time (s)'); ylabel('Kurtosis Value');
xticks([]);

subplot(2,3,5);
plot(linspace(-fs/2, fs/2, NFFT), fftshift(X_power));
set(gca, 'YScale', 'log'); % Apply log scale to enhance visibility
title('FFT-based Spectral Kurtosis'); xlabel('Frequency (Hz)'); ylabel('Power');

subplot(2,3,6);
bar(1:num_bands, kurt_fssk);
title('FSSK Kurtosis per Band'); xlabel('Band'); ylabel('Kurtosis Value');

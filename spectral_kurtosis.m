function [skurt, computation_time] = spectral_kurtosis(signal, fs)
    tic;
    N = length(signal);
    fft_signal = fft(signal);
    power_spectrum = abs(fft_signal).^2 / N;
    mean_power = mean(power_spectrum);
    variance_power = var(power_spectrum);
    skurt = variance_power / (mean_power^2); % Traditional spectral kurtosis formula
    computation_time = toc;
end

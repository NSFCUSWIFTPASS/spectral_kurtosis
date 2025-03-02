% MATLAB Code for ROC Curve Comparison of Full Spectral Kurtosis vs. SDR-Based Frequency Separated Kurtosis
clc; clear; close all;

%% 
% ---------------------------------------------
% --- Genearate noise and interference data ---
% ---------------------------------------------

% create storage vector (Traditional SK value , SDR based FSSK value, RFI_present (1 or 0)]
num_tests = 100;
results = zeros([num_tests, 4]);

% Simulaton Parameters
desired_INR_dB  = 3;    % Interference to noise ratio [dB]
sim_time = 10;          % seconds
fs = 1000;              % sample frequency
T = 1/fs;               % sample period
t = 0:T:sim_time;       % time vectore
f1 = 0.001;             % 
f2 = 10;                % 
impedance = 50;

% Create tests with Interference
% running this in a loop for the sake of my computer's available memory
for ii = 1:num_tests/2 - 1 
    
    % code I stole from Arvind :)
    raw_rfi   = chirp(t,f1,sim_time,f2);
    raw_noise = randn(1,length(t));
    
    % --- Self-normalisation ---
    norm_rfi   = raw_rfi/abs(rms(raw_rfi));
    norm_noise = raw_noise/abs(rms(raw_noise));
    
    N_power     = (rms(norm_noise).^2)./impedance;
    S_power_rfi = (rms(norm_rfi).^2)./impedance;
    
    INR_dB         = 10*log10(S_power_rfi/N_power); 
    INR_multiplier = 10^(desired_INR_dB/20);
    
    rfi_desired = norm_rfi.*INR_multiplier;
    
    S_power_rfi = (rms(rfi_desired).^2)./impedance;
    SNR_dB_check = 10*log10(S_power_rfi/N_power);
    
    rfi = rfi_desired;
    noise = norm_noise;
    
    x = rfi + noise;

    % Compute Kurtosi(s) (plural of Kurtosis?)
    SK_trad_ii = traditional_sk(x);
    fssk_ii    = mean(fssk(x, f1, f2, 5,fs));
    results(ii,:) = [SK_trad_ii, fssk_ii, 1];

    formatSpec = 'RFI Test %d Complete: Trad SK =  %.4f, FSSK = %.4';
    fprintf(formatSpec, SK_trad_ii, fssk_ii)

end

% Tests without interference
for ii = num_tests/2:num_tests

    raw_noise = randn(1,length(t));
    
    norm_noise = raw_noise/abs(rms(raw_noise));
    
    N_power    = (rms(norm_noise).^2)./impedance;
    
    noise = norm_noise;
    
    x = noise;

    % Compute Kurtosi(s) (plural of Kurtosis?)
    SK_trad_ii = traditional_sk(x);
    fssk_ii    = mean(fssk(x, f1, f2, 5,fs));
    results(ii,:) = [SK_trad_ii, fssk_ii, 0];

    formatSpec = 'RFI Test %d Complete: Trad SK =  %.4f, FSSK = %.4';
    fprintf(formatSpec, SK_trad_ii, fssk_ii)

end

%% 
% ----------------------------------
% ------- Compute ROC Curves -------
% ----------------------------------

full_spectral_kurtosis_scores  = results(:,1);
sdr_frequency_separated_scores = results(:,2);
true_labels = results(:,3);

% Compute ROC curves
[fpr_full_sk, tpr_full_sk, ~] = roc_curve(true_labels, full_spectral_kurtosis_scores);
[fpr_sdr_fs, tpr_sdr_fs, ~] = roc_curve(true_labels, sdr_frequency_separated_scores);

% Compute AUC (Area Under Curve)
auc_full_sk = trapz(fpr_full_sk, tpr_full_sk);
auc_sdr_fs = trapz(fpr_sdr_fs, tpr_sdr_fs);

% Plot ROC Curve
figure;
hold on;
plot(fpr_full_sk, tpr_full_sk,'b-', 'LineWidth', 2, 'DisplayName', sprintf('Full Spectral Kurtosis (AUC = %.2f)', auc_full_sk));
plot(fpr_sdr_fs, tpr_sdr_fs, 'r--', 'LineWidth', 2, 'DisplayName', sprintf('SDR-Based Frequency Separated Kurtosis (AUC = %.2f)', auc_sdr_fs));
plot([0 1], [0 1], 'k--', 'DisplayName', 'Random Guess'); % Diagonal reference line

% Formatting the plot
xlabel('False Positive Rate (FPR)');
ylabel('True Positive Rate (TPR)');
title('ROC Curve: Full Spectral Kurtosis vs. SDR-Based Frequency Separated Kurtosis');
legend show;
grid on;
hold off;

%% 
% -------------------------
% ------- Functions -------
% -------------------------

% Function to compute ROC curve
function [fpr, tpr, thresholds] = roc_curve(labels, scores)
    [sorted_scores, idx] = sort(scores, 'descend');
    sorted_labels = labels(idx);
    tpr = cumsum(sorted_labels) / sum(sorted_labels); % True Positive Rate
    fpr = cumsum(~sorted_labels) / sum(~sorted_labels); % False Positive Rate
    thresholds = sorted_scores;
end

% Traditional Spectral Kurtosis (FFT-based)
function spec_kurt = traditional_sk(signal)
    NFFT = 1024;
    X = fft(signal, NFFT);
    X_power = abs(X).^2;
    spec_kurt = kurtosis(X_power);
end

% Frequency Separated Spectral Kurtosis (FSSK, simulated via filtering) 
function kurt_fssk = fssk(signal, f0, f1, num_bands, fs)
    band_edges = linspace(f0, f1, num_bands+1);
    kurt_fssk = zeros(1, num_bands);

    for i = 1:num_bands
        bandpass_filt = designfilt( 'bandpassfir', ...
                                    'FilterOrder', 50, ...
                                    'CutoffFrequency1', band_edges(i), ...
                                    'CutoffFrequency2', band_edges(i+1),...
                                    'SampleRate', fs);
        filtered_signal = filter(bandpass_filt, signal);
        filtered_power = abs(filtered_signal).^2;
        kurt_fssk(i) = kurtosis(filtered_power);
    end
end

% MATLAB Code for ROC Curve Comparison of Full Spectral Kurtosis vs. SDR-Based Frequency Separated Kurtosis
clc; clear; close all;

%% 
% ---------------------------------------------
% --- Genearate noise and interference data ---
% ---------------------------------------------

% create storage vector (Traditional SK value , SDR based FSSK value, RFI_present (1 or 0)]
num_tests = 100;
results = zeros([num_tests, 3]);

% Simulaton Parameters
desired_INR_dB  = 0;    % Interference to noise ratio [dB]
sim_time = 10;          % seconds
fs = 1000;              % sample frequency
T = 1/fs;               % sample period
t = 0:T:sim_time;       % time vectore
f1 = 0.001;             % chirp start freq
f2 = 100;               % chirp end freq
impedance = 50;

subplot_rows = 3;
subplot_columns = 3;

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

    % store the first test for visualization
    if ii == 1
        x_viz = x;
        rfi_viz = rfi;
        noise_viz = noise;
    end

    % Compute Kurtosi(s) (plural of Kurtosis?)
    SK_trad_ii = traditional_sk(x);
    fssk_ii    = mean(fssk(x, f1, f2, 5,fs));
    results(ii,:) = [SK_trad_ii, fssk_ii, 1];

    formatSpec = '\n RFI Test %d Complete: Trad SK =  %.4f, FSSK = %.4f';
    fprintf(formatSpec, ii, SK_trad_ii, fssk_ii)

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

    formatSpec = '\n RFI Test %d Complete: Trad SK = %.4f, FSSK = %.4f';
    fprintf(formatSpec,ii, SK_trad_ii, fssk_ii)

end

%% Plot first test for visualization

x = x_viz;
rfi = rfi_viz;
noise = noise_viz;

% --- Time Domain ---
subplot(subplot_rows,subplot_columns,1)
plot(t,x)
title('Chirp Signal with White Gaussian Noise')
xlabel('Time (s)')
% ylim([amp_limits])

subplot(subplot_rows,subplot_columns,2)
plot(t,noise)
title('White Gaussian Noise')
xlabel('Time (s)')
% ylim([amp_limits])

subplot(subplot_rows,subplot_columns,3)
plot(t,rfi)
title('Chirp')
xlabel('Time (s)')
% ylim([amp_limits])

% --- Histograms ---

bin_factor = 10;
nbins = length(t)/bin_factor;

subplot(subplot_rows,subplot_columns,4)
hist(x, nbins)
title('Hist. of Chirp with AWGN')
dim = [.27 .45 .3 .3];
str = ['Kurtosis: ', num2str(kurtosis(x))];
annotation('textbox',dim,'String',str,'FitBoxToText','on');

subplot(subplot_rows,subplot_columns,5)
hist(noise, nbins)
title('Hist. of AWGN')
dim = [.55 .45 .3 .3];
str = ['Kurtosis: ', num2str(kurtosis(noise))];
annotation('textbox',dim,'String',str,'FitBoxToText','on');

subplot(subplot_rows,subplot_columns,6)
hist(rfi, nbins)
title('Hist. of Noiseless Chirp')
dim = [.80 .45 .3 .3];
str = ['Kurtosis: ', num2str(kurtosis(rfi))];
annotation('textbox',dim,'String',str,'FitBoxToText','on');

% --- Spectrograms ---

num_slices = 20;
window_length = floor(length(t)/num_slices);
overlap = 0; % Nominally, this is floor(0.8*window_length);
kaiser_beta = 5;
cc_unified = [-45 40];
nfft = 512;

% WINDOW = kaiser(window_length,kaiser_beta);
WINDOW = ones(window_length,1); % No-window, basically

subplot(subplot_rows,subplot_columns,7)
[s_x,f_x,t_x] = spectrogram(x,WINDOW,overlap,nfft,fs);
sdb_x = mag2db(abs(s_x));
mesh(t_x,f_x/1000,sdb_x);
cc_x = max(sdb_x(:))+[-60 0];
ax_x = gca;
% ax_x.CLim = cc_x;
ax_x.CLim = cc_unified;
view(2)
colormap(ax_x, 'jet')
colorbar
colorbarHandle1 = colorbar;
ylabel(colorbarHandle1, 'Arbitrary Units');
title('Spectrogram of Chirp Signal with White Gaussian Noise')
xlabel('Time (s)')
ylabel('Freq. (kHz)')

subplot(subplot_rows,subplot_columns,8)
[s_noise,f_noise,t_noise] = spectrogram(noise,WINDOW,overlap,nfft,fs);
sdb_noise = mag2db(abs(s_noise));
mesh(t_noise,f_noise/1000,sdb_noise);
cc_noise = max(sdb_noise(:))+[-60 0];
ax_noise = gca;
% ax_noise.CLim = cc_noise;
ax_noise.CLim = cc_unified;
view(2)
colormap(ax_noise, 'jet')
colorbar
colorbarHandle2 = colorbar;
ylabel(colorbarHandle2, 'Arbitrary Units');
title('Spectrogram of White Gaussian Noise')
xlabel('Time (s)')
ylabel('Frequency (kHz)')

subplot(subplot_rows,subplot_columns,9)
[s_rfi,f_rfi,t_rfi] = spectrogram(rfi,WINDOW,overlap,nfft,fs);
sdb_rfi = mag2db(abs(s_rfi));
mesh(t_rfi,f_rfi/1000,sdb_rfi);
cc_rfi = max(sdb_rfi(:))+[-60 0];
ax_rfi = gca;
% ax_rfi.CLim = cc_rfi;
ax_rfi.CLim = cc_unified;
view(2)
colormap(ax_rfi, 'jet')
colorbar
colorbarHandle3 = colorbar;
ylabel(colorbarHandle3, 'Arbitrary Units');
title('Spectrogram of Chirp Signal only')
xlabel('Time (s)')
ylabel('Frequency (kHz)')

%% 
% ----------------------------------
% ------- Compute ROC Curves -------
% ----------------------------------

full_spectral_kurtosis_scores  = results(:,1);
sdr_frequency_separated_scores = results(:,2);
true_labels = results(:,3);

% histogram of Spectral Kurtocis scores (W/ RFI = Red, No RFI = Blue)
% this will show the seperation of scores w/ and w/out RFI
figure;
nbins = 100;
bin_edges1 = [linspace(0,max(full_spectral_kurtosis_scores),100)];
bin_edges2 = [linspace(0,max(sdr_frequency_separated_scores),100)];

subplot(1,2,1)
histogram(full_spectral_kurtosis_scores(1:49),'BinEdges',bin_edges1)
hold on
histogram(full_spectral_kurtosis_scores(50:end),'BinEdges',bin_edges1)
title('Hist. of Full Spectral Kurtosis Scores')
legend('RFI present', 'No RFI')

subplot(1,2,2)
histogram(sdr_frequency_separated_scores(1:49),'BinEdges',bin_edges2)
hold on
histogram(sdr_frequency_separated_scores(50:end),'BinEdges',bin_edges2)
title('Hist. of Freq-Sep Spectral Kurtosis Scores')
legend('RFI present', 'No RFI')

%%
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

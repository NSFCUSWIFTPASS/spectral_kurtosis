clear
close all
clc

sim_time = 10;

fs = 1000;
T = 1/fs;
t = 0:T:sim_time;
f1 = 0.001;
f2 = 10;

impedance = 50;

desired_SNR_dB = 0;

raw_rfi= chirp(t,f1,sim_time,f2);
raw_noise = randn(1,length(t));
% x = rfi + noise;


% Self-normalisation
%x = x/max(abs(x));
norm_rfi = raw_rfi/abs(rms(raw_rfi));
norm_noise = raw_noise/abs(rms(raw_noise));

% The following two should be 1, since they're both normalised. This then
% gives a default SNR of 0 (equal signal and noise powers).
% rms(norm_noise);
% rms(norm_rfi);

E_noise = sum((raw_noise.^2)./impedance);
P_noise = (raw_noise.^2)./impedance;
E_signal = sum((raw_rfi.^2)./impedance);
P_signal = (raw_rfi.^2)./impedance;

N_power = (rms(norm_noise).^2)./impedance;
S_power_rfi = (rms(norm_rfi).^2)./impedance;

% The objective is to scale the time-series to get the desired SNR. 

SNR_dB = 10*log10(S_power_rfi/N_power);
dispstr = ['Intitially - for the normalised signals, SNR: ', num2str(SNR_dB)];
disp(dispstr)

desired_SNR_dB = 3;

SNR_multiplier = 10^(desired_SNR_dB/20);

rfi_desired = norm_rfi.*SNR_multiplier;

S_power_rfi = (rms(rfi_desired).^2)./impedance;
SNR_dB_check = 10*log10(S_power_rfi/N_power);

dispstr2 = ['Final SNR: ', num2str(SNR_dB_check)];
disp(dispstr2)

rfi = rfi_desired;
noise = norm_noise;

x = rfi + noise;    %  All 5 kurtosis methods must be done on 'x'

figure

title_string = ['SNR (RFI to background noise): ', num2str(SNR_dB_check), ' | Chirp Frequency: ', num2str(f1), ' Hz - ', num2str(f2), ' Hz | Sampling Frequency: ', num2str(fs/1000), ' kHz'];
sgtitle(title_string)

subplot_rows = 6;
subplot_columns = 3;

amp_limits = [-max([max(abs(rfi)), max(abs(noise)), max(abs(x))]), max([max(abs(rfi)), max(abs(noise)), max(abs(x))])];


%% Time domain

subplot(subplot_rows,subplot_columns,1)
plot(t,x)
title('Chirp Signal with White Gaussian Noise')
xlabel('Time (s)')
ylim([amp_limits])

subplot(subplot_rows,subplot_columns,2)
plot(t,noise)
title('White Gaussian Noise')
xlabel('Time (s)')
ylim([amp_limits])

subplot(subplot_rows,subplot_columns,3)
plot(t,rfi)
title('Chirp')
xlabel('Time (s)')
ylim([amp_limits])

% Histograms

bin_factor = 10;
nbins = length(t)/bin_factor;

subplot(subplot_rows,subplot_columns,4)
hist(x, nbins)
title('Histogram of Chirp Signal with White Gaussian Noise')
dim = [.27 .45 .3 .3];
str = ['Kurtosis: ', num2str(kurtosis(x))];
annotation('textbox',dim,'String',str,'FitBoxToText','on');

subplot(subplot_rows,subplot_columns,5)
hist(noise, nbins)
title('Histogram of White Gaussian Noise')
dim = [.55 .45 .3 .3];
str = ['Kurtosis: ', num2str(kurtosis(noise))];
annotation('textbox',dim,'String',str,'FitBoxToText','on');

subplot(subplot_rows,subplot_columns,6)
hist(rfi, nbins)
title('Histogram of Noiseless Chirp Signal')
dim = [.80 .45 .3 .3];
str = ['Kurtosis: ', num2str(kurtosis(rfi))];
annotation('textbox',dim,'String',str,'FitBoxToText','on');

%% Spectrograms

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

%% Spectral Kurtosis (inbuilt MATLAB function)

% Using the spectrogram calculated above as the input

subplot(subplot_rows,subplot_columns,10)
pkurtosis(s_x,fs,f_x, window_length)
title('Chirp Signal with White Gaussian Noise')

subplot(subplot_rows,subplot_columns,11)
pkurtosis(s_noise,fs,f_noise, window_length)
title('White Gaussian Noise only')

subplot(subplot_rows,subplot_columns,12)
pkurtosis(s_rfi,fs,f_rfi, window_length)
title('Chirp only')

%% Manual Spectral Kurtosis (implementing MATLAB function atomically)

% First we have to chop up the single input signal stream into different
% time-windows. Let's use the same window length that we've passed to the
% MATLAB function that calculates the spectrogram.

window_length_manual_SK = floor(length(t)/num_slices);

% Truncating the entire time-domain data to an integer multiple of the
% number of windows and the number of slices (to avoid indexing issues)
x_manual = x(1:(num_slices*window_length_manual_SK));
rfi_manual = rfi(1:(num_slices*window_length_manual_SK));
noise_manual = noise(1:(num_slices*window_length_manual_SK));

% Reshaping the time-domain data into a matrix, where each column is a
% time-slice (i.e. traversing the row in one column advances in time), and the successive columns are successive time slices
x_manual = reshape(x_manual, [window_length_manual_SK, num_slices]);
rfi_manual = reshape(rfi_manual, [window_length_manual_SK, num_slices]);
noise_manual = reshape(noise_manual, [window_length_manual_SK, num_slices]);

% Note that the fft operation works on a matrix, column-wise.
X_w = fft(x_manual); % X_w = fft(x_manual, [], 1) % This forces the Fourier transform to work along the column. But this is the default behaviour of the fft function called on a matrix
RFI_w = fft(rfi_manual);
NOISE_w = fft(noise_manual);

% The columns here refer to the Fourier transform of a short
% time-slice. A single frequency channel would be a row of this matrix.

f_scale = 1;
amp_scale = 1;

%% INSERTING code to APPROPRIATELY SCALE the FFT, and obtain a SINGLE SIDED AMPLITUDE SPECTRUM

if (f_scale == 1)

    L = window_length_manual_SK;
    Fs = fs;
    
    if (amp_scale == 1)
        P2_X = abs(X_w./L);
        P2_RFI = abs(RFI_w./L);
        P2_NOISE = abs(NOISE_w./L);
    else
        P2_X = abs(X_w);
        P2_RFI = abs(RFI_w);
        P2_NOISE = abs(NOISE_w);
    end
    
    P1_X = zeros(((L/2)+1),num_slices);
    P1_RFI = P1_X;
    P1_NOISE = P1_X;
    
    for slice = 1:num_slices
        f = (Fs/L)*(0:(L/2));
    
        P2 = P2_X(:, slice);
        P1 = P2(1:L/2+1);
        P1(2:end-1) = 2*P1(2:end-1);
        P1_X(:,slice) = P1;
    
        P2 = P2_RFI(:, slice);
        P1 = P2(1:L/2+1);
        P1(2:end-1) = 2*P1(2:end-1);
        P1_RFI(:,slice) = P1;
    
        P2 = P2_NOISE(:, slice);
        P1 = P2(1:L/2+1);
        P1(2:end-1) = 2*P1(2:end-1);
        P1_NOISE(:,slice) = P1;
    end
    
    X_w = P1_X;
    RFI_w = P1_RFI;
    NOISE_w = P1_NOISE;

    x_axis_label = 'Frequency (Hz)';
else
    f = (1:window_length);
    x_axis_label = 'Frequency index';
end
%%

% Note that the sum operation works on a matrix, also column-wise by
% default. The second argument (2) to the sum function allows us to override
% the column-wise sum for a row-wise sum. The returned column vector
% contains the post-Fourier transform sum of a single frequency channel,
% for all the frequencies. This is (proportional to) the total energy content in the recorded
% length of the signal, assuming that the time-domain values correspond to
% voltages.

X_w_energy = sum(((abs(X_w)).^2), 2);
RFI_w_energy = sum(((abs(RFI_w)).^2), 2);
NOISE_w_energy = sum(((abs(NOISE_w)).^2), 2);

% Or, using nomenclature from the paper
S1_X = X_w_energy;
S1_RFI = RFI_w_energy;
S1_NOISE = NOISE_w_energy;

S2_X = sum(((abs(X_w)).^4), 2);
S2_RFI = sum(((abs(RFI_w)).^4), 2);
S2_NOISE = sum(((abs(NOISE_w)).^4), 2);

% Computing the Spectral Kurtosis metric now...

M = numel(X_w(1,:));
% This should be equal to the number of slices - i.e. M = num_slices;

SK_X_manual = (S2_X./S1_X) - 2;
SK_RFI_manual = (S2_RFI./S1_RFI) - 2;
SK_NOISE_manual = (S2_NOISE./S1_NOISE) - 2;

subplot(subplot_rows,subplot_columns,13)
plot(f, SK_X_manual)
title('Chirp Signal with White Gaussian Noise [Manual]')
ylabel('Spectral Kurtosis')
xlabel(x_axis_label)

subplot(subplot_rows,subplot_columns,14)
plot(f, SK_NOISE_manual)
title('White Gaussian Noise [Manual]')
ylabel('Spectral Kurtosis')
xlabel(x_axis_label)

subplot(subplot_rows,subplot_columns,15)
plot(f, SK_RFI_manual)
title('Chirp Signal [Manual]')
ylabel('Spectral Kurtosis')
xlabel(x_axis_label)


%% CHIME RFI paper implementation of spectral kurtosis

% First we have to chop up the single input signal stream into different
% time-windows. Let's use the same window length that we've passed to the
% MATLAB function that calculates the spectrogram.

window_length_CHIME_SK = floor(length(t)/20);

% Calculating the number of slices
num_slices = floor(numel(x)/window_length_CHIME_SK);

% Truncating the entire time-domain data to an integer multiple of the
% number of windows and the number of slices (to avoid indexing issues)
x_CHIME = x(1:(num_slices*window_length_CHIME_SK));
rfi_CHIME = rfi(1:(num_slices*window_length_CHIME_SK));
noise_CHIME = noise(1:(num_slices*window_length_CHIME_SK));

% Reshaping the time-domain data into a matrix, where each column is a
% time-slice (i.e. traversing the row in one column advances in time), and the successive columns are successive time slices
x_CHIME = reshape(x_CHIME, [window_length_CHIME_SK, num_slices]);
rfi_CHIME = reshape(rfi_CHIME, [window_length_CHIME_SK, num_slices]);
noise_CHIME = reshape(noise_CHIME, [window_length_CHIME_SK, num_slices]);

% Note that the fft operation works on a matrix, column-wise.
X_w = fft(x_CHIME);
RFI_w = fft(rfi_CHIME);
NOISE_w = fft(noise_CHIME);


% The columns here refer to the Fourier transform of a short
% time-slice. A single frequency channel would be a row of this matrix.

%% INSERTING code to APPROPRIATELY SCALE the FFT, and obtain a SINGLE SIDED AMPLITUDE SPECTRUM
if (f_scale == 1)

    L = window_length_CHIME_SK;
    Fs = fs;
    
    if (amp_scale == 1)
        P2_X = abs(X_w./L);
        P2_RFI = abs(RFI_w./L);
        P2_NOISE = abs(NOISE_w./L);
    else
        P2_X = abs(X_w);
        P2_RFI = abs(RFI_w);
        P2_NOISE = abs(NOISE_w);
    end
    
    P1_X = zeros(((L/2)+1),num_slices);
    P1_RFI = P1_X;
    P1_NOISE = P1_X;
    
    for slice = 1:num_slices
        f = (Fs/L)*(0:(L/2));
    
        P2 = P2_X(:, slice);
        P1 = P2(1:L/2+1);
        P1(2:end-1) = 2*P1(2:end-1);
        P1_X(:,slice) = P1;
    
        P2 = P2_RFI(:, slice);
        P1 = P2(1:L/2+1);
        P1(2:end-1) = 2*P1(2:end-1);
        P1_RFI(:,slice) = P1;
    
        P2 = P2_NOISE(:, slice);
        P1 = P2(1:L/2+1);
        P1(2:end-1) = 2*P1(2:end-1);
        P1_NOISE(:,slice) = P1;
    end
    
    X_w = P1_X;
    RFI_w = P1_RFI;
    NOISE_w = P1_NOISE;

    x_axis_label = 'Frequency (Hz)';
else
    f = (1:window_length);
    x_axis_label = 'Frequency index';
end

%%

% Note that the sum operation works on a matrix, also column-wise by
% default. The second argument (2) to the sum function allows us to override
% the column-wise sum for a row-wise sum. The returned column vector
% contains the post-Fourier transform sum of a single frequency channel,
% for all the frequencies. This is (proportional to) the total energy content in the recorded
% length of the signal, assuming that the time-domain values correspond to
% voltages.

X_w_energy = sum(((abs(X_w)).^2), 2);
RFI_w_energy = sum(((abs(RFI_w)).^2), 2);
NOISE_w_energy = sum(((abs(NOISE_w)).^2), 2);

% Or, using nomenclature from the paper
S1_X = X_w_energy;
S1_RFI = RFI_w_energy;
S1_NOISE = NOISE_w_energy;

S2_X = sum(((abs(X_w)).^4), 2);
S2_RFI = sum(((abs(RFI_w)).^4), 2);
S2_NOISE = sum(((abs(NOISE_w)).^4), 2);

% Computing the Spectral Kurtosis metric now...

M = numel(X_w(1,:));
% This should be equal to the number of slices - i.e. M = num_slices;

SK_X = ((M+1)/(M-1)).*((M*(S2_X./(S1_X.^2)))-1);
SK_RFI = ((M+1)/(M-1)).*((M*(S2_RFI./(S1_RFI.^2)))-1);
SK_NOISE = ((M+1)/(M-1)).*((M*(S2_NOISE./(S1_NOISE.^2)))-1);

subplot(subplot_rows,subplot_columns,16)
plot(f, SK_X)
title('Chirp Signal with White Gaussian Noise [CHIME]')
ylabel('Spectral Kurtosis')
xlabel(x_axis_label)

subplot(subplot_rows,subplot_columns,17)
plot(f, SK_NOISE)
title('White Gaussian Noise [CHIME]')
ylabel('Spectral Kurtosis')
xlabel(x_axis_label)

subplot(subplot_rows,subplot_columns,18)
plot(f, SK_RFI)
title('Chirp Signal [CHIME]')
ylabel('Spectral Kurtosis')
xlabel(x_axis_label)


%% Figuring out the syntax for an APPROPRIATELY SCALED, SINGLE-SIDED AMPLITUDE SPECTRUM

win_index = 1;
num_windows = 1;
x = rfi((((win_index-1)*window_length) + 1):(((win_index-1)*window_length) + (num_windows*window_length)));
L = numel(x);
Fs = fs;

Y = fft(x);
P2 = abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);
f = Fs/L*(0:(L/2));

figure
plot(f,P1,"LineWidth",1) 
title("Single-Sided Amplitude Spectrum of x(t)")
xlabel("f (Hz)")
ylabel("|P1(f)|")

%% Mathematical validation of the filtering in the time-domain (i.e. channelisation that happens prior to the FFT)
% Akin to RF Baseline noise survey that we are doing



%% Matrix FFT debugging
% X_w = fft(x_manual);
% X_w_b = X_w*0;
% for j = 1:num_slices
%     X_w_b(:,j) = fft(x_manual(:,j));
% end
% X_w_a = X_w;
% X_w_c = fft(x_manual,[],1);


%% STFT

% [s_x,f_x,t_x] = stft(x,fs,Window=kaiser(window_length,kaiser_beta),OverlapLength=overlap,FFTLength=nfft);
% [s_noise,f_noise,t_noise] = stft(noise,fs,Window=kaiser(window_length,kaiser_beta),OverlapLength=overlap,FFTLength=nfft);
% [s_rfi,f_rfi,t_rfi] = stft(rfi,fs,Window=kaiser(window_length,kaiser_beta),OverlapLength=overlap,FFTLength=nfft);
kurtosis = spectralKurtosis(x,f)
kurtosis = spectralKurtosis(x,f,Name=Value)
[kurtosis,spread,centroid] = spectralKurtosis(___)
[___,threshold] = spectralKurtosis(___,ConfidenceLevel=p)
spectralKurtosis(___)

fs = 1000;
t = (0:1/fs:10)';
f1 = 300;
f2 = 400;
x = chirp(t,f1,10,f2) + randn(length(t),1);

kurtosis = spectralKurtosis(x,fs);

spectralKurtosis(x,fs)

fs = 1000;
t = (0:1/fs:10)';
f1 = 300;
f2 = 400;
x = chirp(t,f1,10,f2) + randn(length(t),1);

[s,f] = stft(x,fs,FrequencyRange="onesided");
s = abs(s).^2;

kurtosis = spectralKurtosis(s,f);

spectralKurtosis(s,f)

fs = 1000;
t = (0:1/fs:10)';
f1 = 300;
f2 = 400;
x = chirp(t,f1,10,f2) + randn(length(t),1);

kurtosis = spectralKurtosis(x,fs, ...
                      Window=hamming(round(0.05*fs)), ...
                      OverlapLength=round(0.025*fs), ...
                      Range=[62.5,fs/2]);
spectralKurtosis(x,fs, ...
              Window=hamming(round(0.05*fs)), ...
              OverlapLength=round(0.025*fs), ...
              Range=[62.5,fs/2])
fs = 1000;
t = 0:1/fs:10;
f1 = 300;
f2 = 400;

xc = chirp(t,f1,10,f2);
x = xc + randn(1,length(t));

plot(t,x)
title('Chirp Signal with White Gaussian Noise')

[S,F] = pspectrum(x,fs,"spectrogram", ...
    FrequencyResolution=fs/winlen(length(x)),OverlapPercent=80);

[sK95,~,~,thresh95] = spectralKurtosis(S,F,Scaled=false);

plot(F,sK95)
yline(thresh95*[-1 1])
grid
xlabel("Frequency (Hz)")
title("Spectral Kurtosis of Chirp Signal with White Gaussian Noise")

[sK85,~,~,thresh85] = spectralKurtosis(S,F,Scaled=false,ConfidenceLevel=0.85);

plot(F,sK85)
yline(thresh85*[-1 1])
grid
xlabel("Frequency (Hz)")
title("Spectral Kurtosis of Chirp Signal with Noise at Confidence Level of 0.85")

thresh = [thresh95 thresh85]

function y = winlen(x)
    wdiv = 2.^[1 3:7];
    y = ceil(x/wdiv(find(x < 2.^[6 11:14 Inf],1)));
end

"""Signal processing — windows, filters, spectral analysis."""
import pyscivex as sv
import math

# Generate a test signal: 100 Hz + 250 Hz at 1000 Hz sample rate
sr = 1000.0
t = [i / sr for i in range(1024)]
signal = [math.sin(2 * math.pi * 100 * ti) + 0.5 * math.sin(2 * math.pi * 250 * ti) for ti in t]

# Window functions
hann = sv.signal.hann(256)
hamming = sv.signal.hamming(256)
print(f"Hann window: {hann.shape()} samples")
print(f"Hamming window: {hamming.shape()} samples")

# FIR filter design
lp_filter = sv.signal.firwin_lowpass(0.2, 31)
print(f"Lowpass FIR: {lp_filter.shape()} taps")

# Apply filter
b = lp_filter.to_list()
a = [1.0]
filtered = sv.signal.lfilter(b, a, signal)
print(f"Filtered signal: {filtered.shape()} samples")

# Short-time Fourier transform
stft_result = sv.signal.stft(signal, n_fft=256, hop_length=128)
print(f"STFT result keys: {list(stft_result.keys())}")

# Spectrogram
spec = sv.signal.spectrogram(signal, n_fft=256, hop_length=128)
print(f"Spectrogram keys: {list(spec.keys())}")

# Mel spectrogram
mel = sv.signal.mel_spectrogram(signal, sr=sr, n_fft=256, n_mels=40)
print(f"Mel spectrogram shape: {mel.shape()}")

# Peak detection
peaks = sv.signal.find_peaks(signal)
print(f"Found {len(peaks)} peaks")

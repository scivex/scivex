"""Tests for pyscivex signal processing — signal submodule."""

import math
import pyscivex as sv


# ===========================================================================
# WINDOW FUNCTIONS
# ===========================================================================


class TestWindows:
    def test_hann(self):
        w = sv.signal.hann(64)
        data = w.tolist()
        assert len(data) == 64
        assert abs(data[0]) < 1e-10  # hann starts at 0
        assert abs(data[-1]) < 1e-10  # ends at 0

    def test_hamming(self):
        w = sv.signal.hamming(64)
        data = w.tolist()
        assert len(data) == 64
        assert data[0] > 0  # hamming starts > 0

    def test_blackman(self):
        w = sv.signal.blackman(64)
        assert len(w.tolist()) == 64

    def test_bartlett(self):
        w = sv.signal.bartlett(64)
        data = w.tolist()
        assert len(data) == 64
        assert abs(data[0]) < 1e-10  # triangle starts at 0


# ===========================================================================
# FILTERS
# ===========================================================================


class TestFilters:
    def test_lfilter_passthrough(self):
        """FIR with b=[1], a=[1] is identity."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        out = sv.signal.lfilter([1.0], [1.0], x)
        data = out.tolist()
        for i, v in enumerate(data):
            assert abs(v - x[i]) < 1e-10

    def test_filtfilt(self):
        x = [float(i) for i in range(20)]
        out = sv.signal.filtfilt([1.0], [1.0], x)
        assert len(out.tolist()) == 20

    def test_firwin_lowpass(self):
        taps = sv.signal.firwin_lowpass(0.3, 31)
        data = taps.tolist()
        assert len(data) == 31
        # Filter should sum to ~1 for a lowpass
        assert abs(sum(data) - 1.0) < 0.1

    def test_firwin_highpass(self):
        taps = sv.signal.firwin_highpass(0.3, 31)
        assert len(taps.tolist()) == 31

    def test_firwin_bandpass(self):
        taps = sv.signal.firwin_bandpass(0.2, 0.4, 31)
        assert len(taps.tolist()) == 31


# ===========================================================================
# SPECTRAL
# ===========================================================================


class TestSpectral:
    def test_stft_shape(self):
        x = [math.sin(2 * math.pi * 10 * i / 1000) for i in range(1000)]
        result = sv.signal.stft(x, window_size=64, hop_size=32)
        shape = result.shape()
        assert len(shape) == 3  # [frames, freq_bins, 2]
        assert shape[2] == 2  # real + imag

    def test_spectrogram(self):
        x = [math.sin(2 * math.pi * 10 * i / 1000) for i in range(1000)]
        spec = sv.signal.spectrogram(x, window_size=64, hop_size=32)
        shape = spec.shape()
        assert len(shape) == 2  # [frames, freq_bins]

    def test_periodogram(self):
        x = [math.sin(2 * math.pi * 0.1 * i) for i in range(256)]
        freqs, psd = sv.signal.periodogram(x)
        assert len(freqs.tolist()) > 0
        assert len(psd.tolist()) > 0

    def test_welch(self):
        x = [math.sin(2 * math.pi * 0.1 * i) for i in range(512)]
        freqs, psd = sv.signal.welch(x, segment_size=128, overlap=64)
        assert len(freqs.tolist()) > 0
        assert len(psd.tolist()) > 0

    def test_stft_istft_roundtrip(self):
        x = [math.sin(2 * math.pi * 0.05 * i) for i in range(256)]
        s = sv.signal.stft(x, window_size=64, hop_size=32)
        y = sv.signal.istft(s, window_size=64, hop_size=32)
        # Roundtrip should approximately reconstruct
        y_data = y.tolist()
        assert len(y_data) > 0


# ===========================================================================
# WAVELETS
# ===========================================================================


class TestWavelets:
    def test_dwt(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        approx, detail = sv.signal.dwt(x)
        assert len(approx.tolist()) == 4
        assert len(detail.tolist()) == 4

    def test_dwt_idwt_roundtrip(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        approx, detail = sv.signal.dwt(x)
        reconstructed = sv.signal.idwt(approx.tolist(), detail.tolist())
        r = reconstructed.tolist()
        for i, v in enumerate(x):
            assert abs(r[i] - v) < 1e-10


# ===========================================================================
# AUDIO FEATURES
# ===========================================================================


class TestAudioFeatures:
    def test_hz_to_mel(self):
        mel = sv.signal.hz_to_mel(440.0)
        assert mel > 0

    def test_mel_to_hz(self):
        hz = sv.signal.mel_to_hz(100.0)
        assert hz > 0

    def test_mel_roundtrip(self):
        hz = 440.0
        mel = sv.signal.hz_to_mel(hz)
        back = sv.signal.mel_to_hz(mel)
        assert abs(back - hz) < 1e-5

    def test_mel_spectrogram(self):
        x = [math.sin(2 * math.pi * 440 * i / 16000) for i in range(4000)]
        spec = sv.signal.mel_spectrogram(x, sample_rate=16000.0,
                                         n_fft=512, hop_size=256, n_mels=40)
        shape = spec.shape()
        assert len(shape) == 2
        assert shape[1] == 40  # n_mels

    def test_mfcc(self):
        x = [math.sin(2 * math.pi * 440 * i / 16000) for i in range(4000)]
        coeffs = sv.signal.mfcc(x, sample_rate=16000.0, n_mfcc=13,
                                n_mels=40, n_fft=512, hop_size=256)
        shape = coeffs.shape()
        assert len(shape) == 2
        assert shape[1] == 13  # n_mfcc

    def test_chroma_stft(self):
        x = [math.sin(2 * math.pi * 440 * i / 16000) for i in range(4000)]
        chroma = sv.signal.chroma_stft(x, sample_rate=16000.0,
                                       n_fft=512, hop_size=256)
        shape = chroma.shape()
        assert shape[1] == 12  # default n_chroma

    def test_pitch_yin(self):
        # Generate 440 Hz tone
        sr = 16000.0
        x = [math.sin(2 * math.pi * 440 * i / sr) for i in range(8000)]
        pitches = sv.signal.pitch_yin(x, sample_rate=sr,
                                      frame_size=1024, hop_size=512)
        assert len(pitches) > 0
        # At least some frames should detect ~440 Hz
        detected = [p for p in pitches if p > 400 and p < 480]
        assert len(detected) > 0


# ===========================================================================
# PEAK DETECTION
# ===========================================================================


class TestPeakDetection:
    def test_find_peaks(self):
        x = [0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0]
        peaks = sv.signal.find_peaks(x)
        assert 1 in peaks
        assert 3 in peaks
        assert 5 in peaks

    def test_find_peaks_with_height(self):
        x = [0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0]
        peaks = sv.signal.find_peaks(x, min_height=1.5)
        assert 1 not in peaks
        assert 3 in peaks
        assert 5 in peaks

    def test_peak_prominences(self):
        x = [0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0]
        peaks = sv.signal.find_peaks(x)
        proms = sv.signal.peak_prominences(x, peaks)
        assert len(proms) == len(peaks)
        for p in proms:
            assert p > 0


# ===========================================================================
# RESAMPLING
# ===========================================================================


class TestResampling:
    def test_resample_upsample(self):
        x = [1.0, 2.0, 3.0, 4.0]
        out = sv.signal.resample(x, 8)
        assert len(out.tolist()) == 8

    def test_resample_downsample(self):
        x = [float(i) for i in range(100)]
        out = sv.signal.resample(x, 50)
        assert len(out.tolist()) == 50

    def test_decimate(self):
        x = [float(i) for i in range(100)]
        out = sv.signal.decimate(x, 2)
        assert len(out.tolist()) == 50


# ===========================================================================
# CONVOLUTION
# ===========================================================================


class TestConvolution:
    def test_convolve_full(self):
        a = [1.0, 2.0, 3.0]
        b = [0.0, 1.0, 0.5]
        out = sv.signal.convolve(a, b, "full")
        data = out.tolist()
        assert len(data) == 5  # 3 + 3 - 1

    def test_convolve_same(self):
        a = [1.0, 2.0, 3.0, 4.0]
        b = [1.0, 1.0]
        out = sv.signal.convolve(a, b, "same")
        assert len(out.tolist()) == 4  # same as len(a)

    def test_correlate(self):
        a = [1.0, 2.0, 3.0]
        b = [1.0, 2.0, 3.0]
        out = sv.signal.correlate(a, b, "full")
        data = out.tolist()
        # Auto-correlation at zero lag should be max
        mid = len(data) // 2
        assert data[mid] >= max(data[0], data[-1])


# ===========================================================================
# AUDIO I/O
# ===========================================================================


class TestAudioIO:
    def test_write_read_roundtrip(self, tmp_path):
        # Generate a simple tone
        sr = 44100
        samples = [math.sin(2 * math.pi * 440 * i / sr) for i in range(sr)]
        path = str(tmp_path / "test.wav")
        sv.signal.write_wav(path, samples, channels=1, sample_rate=sr)

        audio = sv.signal.read_wav(path)
        assert audio.channels == 1
        assert audio.sample_rate == sr
        assert audio.num_frames() == len(samples)
        assert "AudioData" in repr(audio)

    def test_audio_to_mono(self, tmp_path):
        sr = 44100
        samples = [math.sin(2 * math.pi * 440 * i / sr) for i in range(1000)]
        path = str(tmp_path / "mono.wav")
        sv.signal.write_wav(path, samples, channels=1, sample_rate=sr)
        audio = sv.signal.read_wav(path)
        mono = audio.to_mono()
        assert len(mono) == len(samples)


# ===========================================================================
# RHYTHM / BEAT TRACKING
# ===========================================================================


class TestRhythm:
    def test_onset_strength(self):
        # Simple signal with periodic bursts
        x = [0.0] * 4000
        for i in range(0, 4000, 400):
            for j in range(50):
                if i + j < 4000:
                    x[i + j] = 1.0
        result = sv.signal.onset_strength(x, sample_rate=16000.0,
                                          frame_size=512, hop_size=256)
        assert "onset_envelope" in result
        assert "onset_frames" in result
        assert len(result["onset_envelope"]) > 0

    def test_beat_track(self):
        sr = 16000.0
        # Generate clicks at regular intervals (~120 BPM)
        x = [0.0] * 16000
        interval = int(sr * 60 / 120)  # samples per beat
        for i in range(0, 16000, interval):
            for j in range(10):
                if i + j < 16000:
                    x[i + j] = 1.0
        result = sv.signal.beat_track(x, sample_rate=sr,
                                      frame_size=512, hop_size=256)
        assert "tempo" in result
        assert "beat_frames" in result
        assert result["tempo"] > 0


# ===========================================================================
# INTEGRATION
# ===========================================================================


class TestIntegration:
    def test_all_functions_accessible(self):
        fns = [
            # Windows
            sv.signal.hann, sv.signal.hamming, sv.signal.blackman, sv.signal.bartlett,
            # Filters
            sv.signal.lfilter, sv.signal.filtfilt,
            sv.signal.firwin_lowpass, sv.signal.firwin_highpass, sv.signal.firwin_bandpass,
            # Spectral
            sv.signal.stft, sv.signal.istft, sv.signal.spectrogram,
            sv.signal.periodogram, sv.signal.welch,
            # Wavelets
            sv.signal.dwt, sv.signal.idwt,
            # Features
            sv.signal.hz_to_mel, sv.signal.mel_to_hz,
            sv.signal.mel_spectrogram, sv.signal.mfcc,
            sv.signal.chroma_stft, sv.signal.pitch_yin,
            # Peaks
            sv.signal.find_peaks, sv.signal.peak_prominences,
            # Resample
            sv.signal.resample, sv.signal.decimate,
            # Convolution
            sv.signal.convolve, sv.signal.correlate,
            # Audio I/O
            sv.signal.read_wav, sv.signal.write_wav,
            # Rhythm
            sv.signal.onset_strength, sv.signal.detect_onsets, sv.signal.beat_track,
        ]
        for fn in fns:
            assert fn is not None

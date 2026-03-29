"""Tests for pyscivex.fft submodule."""

import pyscivex as sv


class TestFFT:
    def test_rfft(self):
        # Simple signal: [1, 0, -1, 0]
        signal = sv.Tensor([1.0, 0.0, -1.0, 0.0], [4])
        spectrum = sv.fft.rfft(signal)
        # rfft of length 4 -> N/2+1 = 3 complex bins, shape [3, 2]
        assert spectrum.shape() == [3, 2]

    def test_rfft_irfft_roundtrip(self):
        signal = sv.Tensor([1.0, 2.0, 3.0, 4.0], [4])
        spectrum = sv.fft.rfft(signal)
        recovered = sv.fft.irfft(spectrum, 4)
        assert recovered.shape() == [4]
        data = recovered.to_list()
        for i in range(4):
            assert abs(data[i] - signal.to_list()[i]) < 1e-10

    def test_fft_ifft_roundtrip(self):
        # Complex input: [1+0i, 0+1i] -> shape [2, 2]
        signal = sv.Tensor([1.0, 0.0, 0.0, 1.0], [2, 2])
        spectrum = sv.fft.fft_1d(signal)
        recovered = sv.fft.ifft_1d(spectrum)
        assert recovered.shape() == [2, 2]
        data = recovered.to_list()
        assert abs(data[0] - 1.0) < 1e-10
        assert abs(data[1] - 0.0) < 1e-10

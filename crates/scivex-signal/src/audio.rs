//! WAV audio file reading and writing.
//!
//! Supports:
//! - 8-bit unsigned PCM
//! - 16-bit signed PCM
//! - 32-bit signed PCM
//! - 32-bit IEEE float
//!
//! All data is returned as `Vec<f64>` samples normalised to [-1.0, 1.0].

use crate::error::{Result, SignalError};

/// Audio data loaded from a WAV file.
#[derive(Debug, Clone)]
pub struct AudioData {
    /// Interleaved sample data, normalised to [-1.0, 1.0].
    pub samples: Vec<f64>,
    /// Number of channels (1 = mono, 2 = stereo).
    pub channels: u16,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Bits per sample in the original file.
    pub bits_per_sample: u16,
}

impl AudioData {
    /// Number of frames (samples per channel).
    pub fn num_frames(&self) -> usize {
        if self.channels == 0 {
            return 0;
        }
        self.samples.len() / self.channels as usize
    }

    /// Extract a single channel (0-indexed) as a `Vec<f64>`.
    pub fn channel(&self, ch: usize) -> Result<Vec<f64>> {
        let nc = self.channels as usize;
        if ch >= nc {
            return Err(SignalError::InvalidParameter {
                name: "channel",
                reason: "channel index out of range",
            });
        }
        Ok(self.samples.iter().skip(ch).step_by(nc).copied().collect())
    }

    /// Convert stereo to mono by averaging channels.
    pub fn to_mono(&self) -> Vec<f64> {
        let nc = self.channels as usize;
        if nc <= 1 {
            return self.samples.clone();
        }
        let frames = self.num_frames();
        let mut mono = Vec::with_capacity(frames);
        for i in 0..frames {
            let mut sum = 0.0;
            for ch in 0..nc {
                sum += self.samples[i * nc + ch];
            }
            mono.push(sum / nc as f64);
        }
        mono
    }
}

/// Read a WAV file from bytes.
///
/// Supports PCM (8/16/32-bit) and IEEE float (32-bit) formats.
pub fn read_wav(data: &[u8]) -> Result<AudioData> {
    if data.len() < 44 {
        return Err(SignalError::InvalidParameter {
            name: "data",
            reason: "WAV file too small",
        });
    }

    // RIFF header
    if &data[0..4] != b"RIFF" || &data[8..12] != b"WAVE" {
        return Err(SignalError::InvalidParameter {
            name: "data",
            reason: "not a valid WAV file",
        });
    }

    // Find fmt chunk
    let mut pos = 12;
    let mut audio_format: u16 = 0;
    let mut channels: u16 = 0;
    let mut sample_rate: u32 = 0;
    let mut bits_per_sample: u16 = 0;
    let mut found_fmt = false;

    while pos + 8 <= data.len() {
        let chunk_id = &data[pos..pos + 4];
        let chunk_size =
            u32::from_le_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]])
                as usize;

        if chunk_id == b"fmt " {
            if pos + 8 + chunk_size > data.len() || chunk_size < 16 {
                return Err(SignalError::InvalidParameter {
                    name: "data",
                    reason: "invalid fmt chunk",
                });
            }
            let fmt = &data[pos + 8..];
            audio_format = u16::from_le_bytes([fmt[0], fmt[1]]);
            channels = u16::from_le_bytes([fmt[2], fmt[3]]);
            sample_rate = u32::from_le_bytes([fmt[4], fmt[5], fmt[6], fmt[7]]);
            // skip byte_rate (4) and block_align (2)
            bits_per_sample = u16::from_le_bytes([fmt[14], fmt[15]]);
            found_fmt = true;
        }

        if chunk_id == b"data" && found_fmt {
            let sample_data = &data[pos + 8..pos + 8 + chunk_size.min(data.len() - pos - 8)];
            let samples = decode_samples(sample_data, audio_format, bits_per_sample)?;
            return Ok(AudioData {
                samples,
                channels,
                sample_rate,
                bits_per_sample,
            });
        }

        pos += 8 + chunk_size;
        // Chunks are word-aligned
        if !chunk_size.is_multiple_of(2) {
            pos += 1;
        }
    }

    Err(SignalError::InvalidParameter {
        name: "data",
        reason: "no data chunk found in WAV file",
    })
}

/// Write audio data to WAV format bytes (16-bit PCM).
pub fn write_wav(audio: &AudioData) -> Result<Vec<u8>> {
    write_wav_bits(audio, 16)
}

/// Write audio data to WAV format bytes with specified bit depth.
pub fn write_wav_bits(audio: &AudioData, bits: u16) -> Result<Vec<u8>> {
    if audio.samples.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    if bits != 8 && bits != 16 && bits != 32 {
        return Err(SignalError::InvalidParameter {
            name: "bits",
            reason: "must be 8, 16, or 32",
        });
    }

    let bytes_per_sample = (bits / 8) as usize;
    let block_align = (audio.channels as usize) * bytes_per_sample;
    let data_size = audio.samples.len() * bytes_per_sample;
    let file_size = 36 + data_size;

    let mut buf = Vec::with_capacity(file_size + 8);

    // RIFF header
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&(file_size as u32).to_le_bytes());
    buf.extend_from_slice(b"WAVE");

    // fmt chunk
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes()); // chunk size
    buf.extend_from_slice(&1u16.to_le_bytes()); // PCM format
    buf.extend_from_slice(&audio.channels.to_le_bytes());
    buf.extend_from_slice(&audio.sample_rate.to_le_bytes());
    let byte_rate = (audio.sample_rate as usize * block_align) as u32;
    buf.extend_from_slice(&byte_rate.to_le_bytes());
    buf.extend_from_slice(&(block_align as u16).to_le_bytes());
    buf.extend_from_slice(&bits.to_le_bytes());

    // data chunk
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&(data_size as u32).to_le_bytes());

    // Encode samples
    for &s in &audio.samples {
        match bits {
            8 => {
                // 8-bit unsigned: 0..255, 128 = silence
                let val = ((s * 127.0) + 128.0).clamp(0.0, 255.0) as u8;
                buf.push(val);
            }
            16 => {
                let val = (s * 32767.0).clamp(-32768.0, 32767.0) as i16;
                buf.extend_from_slice(&val.to_le_bytes());
            }
            32 => {
                let val = (s * 2_147_483_647.0).clamp(-2_147_483_648.0, 2_147_483_647.0) as i32;
                buf.extend_from_slice(&val.to_le_bytes());
            }
            _ => unreachable!(),
        }
    }

    Ok(buf)
}

fn decode_samples(data: &[u8], format: u16, bits: u16) -> Result<Vec<f64>> {
    match (format, bits) {
        // PCM 8-bit unsigned
        (1, 8) => Ok(data
            .iter()
            .map(|&b| (f64::from(b) - 128.0) / 128.0)
            .collect()),
        // PCM 16-bit signed
        (1, 16) => {
            if !data.len().is_multiple_of(2) {
                return Err(SignalError::InvalidParameter {
                    name: "data",
                    reason: "16-bit data not aligned",
                });
            }
            Ok(data
                .chunks_exact(2)
                .map(|c| {
                    let val = i16::from_le_bytes([c[0], c[1]]);
                    f64::from(val) / 32768.0
                })
                .collect())
        }
        // PCM 32-bit signed
        (1, 32) => {
            if !data.len().is_multiple_of(4) {
                return Err(SignalError::InvalidParameter {
                    name: "data",
                    reason: "32-bit data not aligned",
                });
            }
            Ok(data
                .chunks_exact(4)
                .map(|c| {
                    let val = i32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                    f64::from(val) / 2_147_483_648.0
                })
                .collect())
        }
        // IEEE float 32-bit
        (3, 32) => {
            if !data.len().is_multiple_of(4) {
                return Err(SignalError::InvalidParameter {
                    name: "data",
                    reason: "float data not aligned",
                });
            }
            Ok(data
                .chunks_exact(4)
                .map(|c| {
                    let val = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                    f64::from(val)
                })
                .collect())
        }
        _ => Err(SignalError::InvalidParameter {
            name: "format",
            reason: "unsupported WAV format (only PCM 8/16/32 and IEEE float 32 supported)",
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sine_audio(freq: f64, sample_rate: u32, duration_secs: f64) -> AudioData {
        let sr = f64::from(sample_rate);
        let num_samples = (sr * duration_secs) as usize;
        let samples: Vec<f64> = (0..num_samples)
            .map(|i| {
                let t = i as f64 / sr;
                (2.0 * std::f64::consts::PI * freq * t).sin()
            })
            .collect();
        AudioData {
            samples,
            channels: 1,
            sample_rate,
            bits_per_sample: 16,
        }
    }

    #[test]
    fn test_wav_roundtrip_16bit() {
        let audio = make_sine_audio(440.0, 44100, 0.01);
        let wav_bytes = write_wav(&audio).unwrap();
        let loaded = read_wav(&wav_bytes).unwrap();

        assert_eq!(loaded.channels, 1);
        assert_eq!(loaded.sample_rate, 44100);
        assert_eq!(loaded.bits_per_sample, 16);
        assert_eq!(loaded.samples.len(), audio.samples.len());

        // 16-bit quantisation introduces some error
        for (a, b) in audio.samples.iter().zip(loaded.samples.iter()) {
            assert!((a - b).abs() < 0.001, "sample mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_wav_roundtrip_8bit() {
        let audio = make_sine_audio(440.0, 8000, 0.01);
        let wav_bytes = write_wav_bits(&audio, 8).unwrap();
        let loaded = read_wav(&wav_bytes).unwrap();

        assert_eq!(loaded.channels, 1);
        assert_eq!(loaded.sample_rate, 8000);
        assert_eq!(loaded.bits_per_sample, 8);
        assert_eq!(loaded.samples.len(), audio.samples.len());

        // 8-bit has lower precision
        for (a, b) in audio.samples.iter().zip(loaded.samples.iter()) {
            assert!((a - b).abs() < 0.02, "sample mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_wav_roundtrip_32bit() {
        let audio = make_sine_audio(440.0, 44100, 0.01);
        let wav_bytes = write_wav_bits(&audio, 32).unwrap();
        let loaded = read_wav(&wav_bytes).unwrap();

        assert_eq!(loaded.bits_per_sample, 32);
        for (a, b) in audio.samples.iter().zip(loaded.samples.iter()) {
            assert!((a - b).abs() < 1e-6, "sample mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_stereo_wav() {
        let samples: Vec<f64> = (0..100)
            .map(|i| if i % 2 == 0 { 0.5 } else { -0.5 })
            .collect();
        let audio = AudioData {
            samples,
            channels: 2,
            sample_rate: 44100,
            bits_per_sample: 16,
        };
        let wav_bytes = write_wav(&audio).unwrap();
        let loaded = read_wav(&wav_bytes).unwrap();

        assert_eq!(loaded.channels, 2);
        assert_eq!(loaded.num_frames(), 50);
    }

    #[test]
    fn test_channel_extraction() {
        let samples = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let audio = AudioData {
            samples,
            channels: 2,
            sample_rate: 44100,
            bits_per_sample: 16,
        };

        let left = audio.channel(0).unwrap();
        let right = audio.channel(1).unwrap();
        assert_eq!(left.len(), 3);
        assert_eq!(right.len(), 3);
        assert!((left[0] - 0.1).abs() < 1e-10);
        assert!((right[0] - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_to_mono() {
        let samples = vec![0.2, 0.4, 0.6, 0.8];
        let audio = AudioData {
            samples,
            channels: 2,
            sample_rate: 44100,
            bits_per_sample: 16,
        };

        let mono = audio.to_mono();
        assert_eq!(mono.len(), 2);
        assert!((mono[0] - 0.3).abs() < 1e-10);
        assert!((mono[1] - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_invalid_wav_too_small() {
        assert!(read_wav(&[0u8; 10]).is_err());
    }

    #[test]
    fn test_invalid_wav_wrong_header() {
        let mut data = vec![0u8; 44];
        data[0..4].copy_from_slice(b"XXXX");
        assert!(read_wav(&data).is_err());
    }

    #[test]
    fn test_empty_audio_write() {
        let audio = AudioData {
            samples: vec![],
            channels: 1,
            sample_rate: 44100,
            bits_per_sample: 16,
        };
        assert!(write_wav(&audio).is_err());
    }

    #[test]
    fn test_num_frames() {
        let audio = AudioData {
            samples: vec![0.0; 100],
            channels: 2,
            sample_rate: 44100,
            bits_per_sample: 16,
        };
        assert_eq!(audio.num_frames(), 50);
    }

    #[test]
    fn test_channel_out_of_range() {
        let audio = AudioData {
            samples: vec![0.0; 10],
            channels: 1,
            sample_rate: 44100,
            bits_per_sample: 16,
        };
        assert!(audio.channel(1).is_err());
    }
}

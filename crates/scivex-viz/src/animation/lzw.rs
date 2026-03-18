//! LZW compression for GIF89a.
//!
//! Implements variable-width LZW encoding as required by the GIF specification.

/// LZW-compress data for GIF encoding.
///
/// - `min_code_size`: minimum code size (typically 8 for 256-color images,
///   equal to ceil(log2(palette_size)) but at least 2).
/// - `data`: palette-indexed pixel data.
///
/// Returns the compressed byte stream (sub-blocks are handled by the caller).
pub fn lzw_encode(min_code_size: u8, data: &[u8]) -> Vec<u8> {
    let clear_code = 1u32 << min_code_size;
    let eoi_code = clear_code + 1;
    let first_free = clear_code + 2;

    let mut output = BitWriter::new();
    let mut code_size = u32::from(min_code_size) + 1;
    let max_code_limit = 4096u32;

    // Dictionary: map (prefix_code, byte) -> code.
    let mut dict = Dictionary::new();
    let mut next_code = first_free;

    // Emit clear code to start.
    output.write_bits(clear_code, code_size);

    // Initialize dictionary with single-byte entries.
    dict.clear(clear_code as usize);

    let mut prefix: Option<u32> = None;

    for &byte in data {
        if let Some(p) = prefix {
            if let Some(code) = dict.get(p, byte) {
                prefix = Some(code);
            } else {
                // Emit prefix code.
                output.write_bits(p, code_size);

                // Add new entry if room.
                if next_code < max_code_limit {
                    dict.insert(p, byte, next_code);
                    // Increase code size if needed.
                    if next_code >= (1 << code_size) {
                        code_size += 1;
                    }
                    next_code += 1;
                } else {
                    // Table full — emit clear code and reset.
                    output.write_bits(clear_code, code_size);
                    dict.clear(clear_code as usize);
                    next_code = first_free;
                    code_size = u32::from(min_code_size) + 1;
                }

                prefix = Some(u32::from(byte));
            }
        } else {
            prefix = Some(u32::from(byte));
        }
    }

    // Emit remaining prefix.
    if let Some(p) = prefix {
        output.write_bits(p, code_size);
    }

    // Emit end-of-information code.
    output.write_bits(eoi_code, code_size);

    output.finish()
}

// ---------------------------------------------------------------------------
// Dictionary
// ---------------------------------------------------------------------------

/// Simple hash-based LZW dictionary.
struct Dictionary {
    // Use a flat hash table: key = (prefix, byte) -> code.
    entries: Vec<DictEntry>,
    mask: usize,
}

#[derive(Clone, Copy)]
struct DictEntry {
    prefix: u32,
    byte: u8,
    code: u32,
    occupied: bool,
}

impl Dictionary {
    fn new() -> Self {
        // Start with a reasonable table size.
        let size = 8192;
        Self {
            entries: vec![
                DictEntry {
                    prefix: 0,
                    byte: 0,
                    code: 0,
                    occupied: false,
                };
                size
            ],
            mask: size - 1,
        }
    }

    fn clear(&mut self, _clear_code: usize) {
        for e in &mut self.entries {
            e.occupied = false;
        }
    }

    fn hash(prefix: u32, byte: u8) -> usize {
        let key = (u64::from(prefix) << 8) | u64::from(byte);
        // FNV-1a hash.
        let mut h = 0xcbf2_9ce4_8422_2325_u64;
        for b in key.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
        h as usize
    }

    fn get(&self, prefix: u32, byte: u8) -> Option<u32> {
        let mut idx = Self::hash(prefix, byte) & self.mask;
        loop {
            let e = &self.entries[idx];
            if !e.occupied {
                return None;
            }
            if e.prefix == prefix && e.byte == byte {
                return Some(e.code);
            }
            idx = (idx + 1) & self.mask;
        }
    }

    fn insert(&mut self, prefix: u32, byte: u8, code: u32) {
        let mut idx = Self::hash(prefix, byte) & self.mask;
        loop {
            let e = &self.entries[idx];
            if !e.occupied {
                self.entries[idx] = DictEntry {
                    prefix,
                    byte,
                    code,
                    occupied: true,
                };
                return;
            }
            idx = (idx + 1) & self.mask;
        }
    }
}

// ---------------------------------------------------------------------------
// BitWriter
// ---------------------------------------------------------------------------

/// Packs variable-width codes into a byte stream (LSB-first).
struct BitWriter {
    buffer: u32,
    bits_in_buffer: u32,
    output: Vec<u8>,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            buffer: 0,
            bits_in_buffer: 0,
            output: Vec::with_capacity(4096),
        }
    }

    fn write_bits(&mut self, code: u32, code_size: u32) {
        self.buffer |= code << self.bits_in_buffer;
        self.bits_in_buffer += code_size;
        while self.bits_in_buffer >= 8 {
            self.output.push((self.buffer & 0xFF) as u8);
            self.buffer >>= 8;
            self.bits_in_buffer -= 8;
        }
    }

    fn finish(mut self) -> Vec<u8> {
        if self.bits_in_buffer > 0 {
            self.output.push((self.buffer & 0xFF) as u8);
        }
        self.output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lzw_encode_basic() {
        // Encode a simple repeated pattern.
        let data: Vec<u8> = vec![0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0];
        let compressed = lzw_encode(2, &data);
        // Should produce valid output (not empty).
        assert!(!compressed.is_empty());
        // Should be smaller or equal to uncompressed + overhead.
        assert!(compressed.len() <= data.len() + 10);
    }

    #[test]
    fn lzw_encode_single_byte() {
        let data = vec![0];
        let compressed = lzw_encode(2, &data);
        assert!(!compressed.is_empty());
    }

    #[test]
    fn lzw_encode_all_same() {
        // Highly compressible.
        let data = vec![5u8; 1000];
        let compressed = lzw_encode(8, &data);
        assert!(compressed.len() < data.len());
    }

    #[test]
    fn lzw_encode_empty() {
        let data: Vec<u8> = vec![];
        let compressed = lzw_encode(2, &data);
        // Should still have clear code + EOI.
        assert!(!compressed.is_empty());
    }
}

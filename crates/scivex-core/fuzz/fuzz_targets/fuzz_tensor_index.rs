#![no_main]

use libfuzzer_sys::fuzz_target;
use scivex_core::Tensor;

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }

    // Use first two bytes for shape, rest for indices
    let rows = (data[0] as usize % 20) + 1;
    let cols = (data[1] as usize % 20) + 1;
    let n = rows * cols;

    // Create a tensor with deterministic data
    let tensor_data: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let t = match Tensor::from_vec(tensor_data, vec![rows, cols]) {
        Ok(t) => t,
        Err(_) => return,
    };

    // Try indexing with fuzzed indices
    for chunk in data[2..].chunks(2) {
        if chunk.len() < 2 {
            break;
        }
        let i = chunk[0] as usize;
        let j = chunk[1] as usize;
        // Should never panic, only return Ok or Err
        let _ = t.get(&[i, j]);
    }

    // Try reshape with fuzzed dimensions
    if data.len() >= 6 {
        let d1 = data[2] as usize + 1;
        let d2 = data[3] as usize + 1;
        let _ = t.reshape(vec![d1, d2]);
    }
});

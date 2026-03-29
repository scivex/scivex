#![no_main]

use libfuzzer_sys::fuzz_target;
use scivex_core::Tensor;

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }

    // Create a tensor of size data[0]
    let n = (data[0] as usize % 100) + 1;
    let tensor_data: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let t = Tensor::from_vec(tensor_data, vec![n]).unwrap();

    // Try reshape with dimensions derived from remaining bytes
    for chunk in data[1..].chunks(2) {
        let ndims = (chunk[0] as usize % 4) + 1;
        let mut shape = Vec::with_capacity(ndims);
        for i in 0..ndims {
            let idx = if i < chunk.len() { chunk[i] } else { 1 };
            shape.push((idx as usize % 50) + 1);
        }
        // Should never panic
        let _ = t.reshape(shape);
    }

    // Try various operations that should be safe
    let _ = t.sum();
    let _ = t.mean();
    let _ = t.flatten();
    let _ = t.map(|x| x * 2.0);
});

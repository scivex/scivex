#![no_main]

use libfuzzer_sys::fuzz_target;
use scivex_core::Tensor;

fuzz_target!(|data: &[u8]| {
    // Treat input as a UTF-8 string for einsum notation
    let notation = match std::str::from_utf8(data) {
        Ok(s) => s,
        Err(_) => return,
    };

    // Limit input length to avoid excessive computation
    if notation.len() > 100 {
        return;
    }

    // Create small test tensors
    let a = Tensor::from_vec(vec![1.0_f64; 9], vec![3, 3]).unwrap();
    let b = Tensor::from_vec(vec![1.0_f64; 9], vec![3, 3]).unwrap();

    // Try parsing arbitrary einsum notation — should never panic
    let _ = scivex_core::einsum::einsum(notation, &[&a, &b]);
    let _ = scivex_core::einsum::einsum(notation, &[&a]);
});

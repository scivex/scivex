#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Treat input as CSV content
    let csv_str = match std::str::from_utf8(data) {
        Ok(s) => s,
        Err(_) => return,
    };

    // Limit size to avoid excessive computation
    if csv_str.len() > 10_000 {
        return;
    }

    // Try parsing arbitrary CSV content — should never panic
    let cursor = std::io::Cursor::new(csv_str.as_bytes());
    let _ = scivex_io::csv::read_csv(cursor);
});

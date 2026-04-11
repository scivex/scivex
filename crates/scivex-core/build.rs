fn main() {
    // Link system BLAS when blas-backend feature is enabled.
    #[cfg(feature = "blas-backend")]
    {
        let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
        match target_os.as_str() {
            "macos" | "ios" => {
                // Apple Accelerate framework provides CBLAS.
                println!("cargo:rustc-link-lib=framework=Accelerate");
            }
            "linux" => {
                // OpenBLAS provides CBLAS on most Linux distros.
                println!("cargo:rustc-link-lib=openblas");
            }
            "windows" => {
                // On Windows, try OpenBLAS.
                println!("cargo:rustc-link-lib=openblas");
            }
            _ => {}
        }
    }
}

//! Cloud storage URL-based reading.
//!
//! Provides a unified interface for reading objects from cloud storage (AWS S3,
//! Google Cloud Storage, Azure Blob Storage) as well as local files, all behind
//! a common URL scheme.
//!
//! # Supported URL schemes
//!
//! | Scheme | Provider | Example |
//! |--------|----------|---------|
//! | `s3://` | Amazon S3 | `s3://my-bucket/path/to/file.csv` |
//! | `gs://` | Google Cloud Storage | `gs://my-bucket/data.parquet` |
//! | `az://` | Azure Blob Storage | `az://container/blob` |
//! | `wasbs://` | Azure (WASBS) | `wasbs://container@account.blob.core.windows.net/path` |
//! | `file://` | Local filesystem | `file:///tmp/data.csv` |
//! | *(none)* | Local filesystem | `/home/user/data.csv` |
//!
//! # Feature gate
//!
//! This module is gated behind the `cloud` feature flag.
//!
//! # Example
//!
//! ```rust,no_run
//! use scivex_io::cloud::{CloudPath, CloudConfig, cloud_read_bytes, StdHttpClient};
//!
//! let config = CloudConfig::from_env();
//! let http = StdHttpClient;
//! let data = cloud_read_bytes("s3://my-bucket/data.csv", &config, &http).unwrap();
//! ```

#[allow(dead_code)]
mod auth;
mod parse;
pub mod s3;
pub mod sha256;

use std::io::Read;
use std::path::PathBuf;

use crate::error::{IoError, Result};

// ───────────────────────────── Public types ─────────────────────────────

/// A parsed cloud storage path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CloudPath {
    /// Amazon S3 object.
    S3 {
        /// Bucket name.
        bucket: String,
        /// Object key (path within bucket).
        key: String,
        /// Optional AWS region override.
        region: Option<String>,
    },
    /// Google Cloud Storage object.
    Gcs {
        /// Bucket name.
        bucket: String,
        /// Object key.
        key: String,
    },
    /// Azure Blob Storage object.
    Azure {
        /// Storage account name (may be empty for `az://` scheme — supply
        /// via [`CloudConfig`] in that case).
        account: String,
        /// Container name.
        container: String,
        /// Blob path.
        blob: String,
    },
    /// Local filesystem path.
    Local(PathBuf),
}

impl CloudPath {
    /// Parse a URL string into a [`CloudPath`].
    ///
    /// See [module-level docs](self) for supported schemes.
    pub fn parse(url: &str) -> Result<Self> {
        parse::parse_cloud_url(url)
    }
}

/// Configuration for cloud storage access.
///
/// Fields are resolved in order: explicit values first, then environment
/// variables. See [`auth`] for resolution details.
#[derive(Debug, Clone, Default)]
pub struct CloudConfig {
    // AWS
    /// AWS access key ID.
    pub aws_access_key_id: Option<String>,
    /// AWS secret access key.
    pub aws_secret_access_key: Option<String>,
    /// AWS region (defaults to `us-east-1`).
    pub aws_region: Option<String>,
    /// AWS session token (for temporary credentials).
    pub aws_session_token: Option<String>,

    // GCS
    /// Path to GCS service account key JSON file.
    pub gcs_service_account_key: Option<String>,

    // Azure
    /// Azure storage account name.
    pub azure_storage_account: Option<String>,
    /// Azure storage account key.
    pub azure_storage_key: Option<String>,
    /// Azure SAS token.
    pub azure_sas_token: Option<String>,
}

impl CloudConfig {
    /// Create a config populated entirely from environment variables.
    ///
    /// See [`auth`] for which environment variables are checked.
    pub fn from_env() -> Self {
        Self {
            aws_access_key_id: std::env::var("AWS_ACCESS_KEY_ID").ok(),
            aws_secret_access_key: std::env::var("AWS_SECRET_ACCESS_KEY").ok(),
            aws_region: std::env::var("AWS_DEFAULT_REGION")
                .ok()
                .or_else(|| std::env::var("AWS_REGION").ok()),
            aws_session_token: std::env::var("AWS_SESSION_TOKEN").ok(),
            gcs_service_account_key: std::env::var("GOOGLE_APPLICATION_CREDENTIALS").ok(),
            azure_storage_account: std::env::var("AZURE_STORAGE_ACCOUNT").ok(),
            azure_storage_key: std::env::var("AZURE_STORAGE_KEY").ok(),
            azure_sas_token: std::env::var("AZURE_SAS_TOKEN").ok(),
        }
    }
}

// ───────────────────────────── HTTP trait ─────────────────────────────

/// Minimal HTTP client trait for cloud storage operations.
///
/// Implementations perform a blocking HTTP GET and return the response body.
/// This trait allows swapping in different HTTP backends (plain TCP for
/// testing, TLS-capable client for production, etc.).
pub trait HttpClient {
    /// Perform an HTTP GET request.
    ///
    /// `headers` contains `(name, value)` pairs to include in the request.
    /// Returns the response body bytes on success.
    fn get(&self, url: &str, headers: &[(String, String)]) -> Result<Vec<u8>>;
}

/// A minimal HTTP/1.1 client using `std::net::TcpStream`.
///
/// Supports **plain HTTP only** (no TLS). Suitable for testing against local
/// mock servers (e.g., MinIO, LocalStack).
///
/// For production use with HTTPS endpoints, provide a TLS-capable
/// [`HttpClient`] implementation instead.
pub struct StdHttpClient;

impl HttpClient for StdHttpClient {
    fn get(&self, url: &str, headers: &[(String, String)]) -> Result<Vec<u8>> {
        use std::fmt::Write as _;

        let (host, port, path) = parse_http_url(url)?;

        // Build the request
        let mut request = format!("GET {path} HTTP/1.1\r\n");
        let _ = write!(request, "Host: {host}\r\n");
        request.push_str("Connection: close\r\n");

        for (name, value) in headers {
            // Skip host (we already added it) and authorization goes through
            if name.eq_ignore_ascii_case("host") {
                continue;
            }
            let _ = write!(request, "{name}: {value}\r\n");
        }
        request.push_str("\r\n");

        // Connect
        let addr = format!("{host}:{port}");
        let mut stream = std::net::TcpStream::connect(&addr).map_err(|e| {
            IoError::Io(std::io::Error::new(
                e.kind(),
                format!("failed to connect to {addr}: {e}"),
            ))
        })?;

        // Send request
        std::io::Write::write_all(&mut stream, request.as_bytes())?;

        // Read full response
        let mut response = Vec::new();
        stream.read_to_end(&mut response)?;

        // Parse HTTP response — find the body after \r\n\r\n
        let header_end = find_header_end(&response).ok_or_else(|| {
            IoError::FormatError("invalid HTTP response: no header terminator".to_string())
        })?;

        let status_line_end = response
            .iter()
            .position(|&b| b == b'\r')
            .unwrap_or(header_end);
        let status_line = String::from_utf8_lossy(&response[..status_line_end]);

        // Check status code
        let status_code = parse_status_code(&status_line)?;
        if !(200..300).contains(&status_code) {
            let body = String::from_utf8_lossy(&response[header_end..]);
            return Err(IoError::FormatError(format!("HTTP {status_code}: {body}")));
        }

        Ok(response[header_end..].to_vec())
    }
}

/// Parse a simple HTTP URL into (host, port, path).
fn parse_http_url(url: &str) -> Result<(String, u16, String)> {
    let (scheme, rest) = if let Some(rest) = url.strip_prefix("https://") {
        ("https", rest)
    } else if let Some(rest) = url.strip_prefix("http://") {
        ("http", rest)
    } else {
        return Err(IoError::FormatError(format!(
            "unsupported URL scheme in: {url}"
        )));
    };

    if scheme == "https" {
        return Err(IoError::FormatError(
            "StdHttpClient does not support HTTPS — use a TLS-capable HttpClient \
             implementation for production cloud access"
                .to_string(),
        ));
    }

    let (host_port, path) = match rest.find('/') {
        Some(pos) => (&rest[..pos], &rest[pos..]),
        None => (rest, "/"),
    };

    let (host, port) = match host_port.find(':') {
        Some(pos) => {
            let port_str = &host_port[pos + 1..];
            let port: u16 = port_str
                .parse()
                .map_err(|_| IoError::FormatError(format!("invalid port: {port_str}")))?;
            (host_port[..pos].to_string(), port)
        }
        None => (host_port.to_string(), 80),
    };

    Ok((host, port, path.to_string()))
}

/// Find the position after `\r\n\r\n` in the response.
fn find_header_end(data: &[u8]) -> Option<usize> {
    data.windows(4)
        .position(|w| w == b"\r\n\r\n")
        .map(|pos| pos + 4)
}

/// Parse the HTTP status code from a status line like "HTTP/1.1 200 OK".
fn parse_status_code(status_line: &str) -> Result<u16> {
    let parts: Vec<&str> = status_line.split_whitespace().collect();
    if parts.len() < 2 {
        return Err(IoError::FormatError(format!(
            "invalid HTTP status line: {status_line}"
        )));
    }
    parts[1]
        .parse()
        .map_err(|_| IoError::FormatError(format!("invalid HTTP status code in: {status_line}")))
}

// ───────────────────────────── cloud_read_bytes ─────────────────────────────

/// Download a cloud object (or read a local file) and return its contents
/// as bytes.
///
/// The URL is parsed using [`CloudPath::parse`], and the appropriate backend
/// is used to fetch the data. An [`HttpClient`] implementation must be
/// provided for remote storage; local paths are read directly via `std::fs`.
///
/// # Errors
///
/// Returns an error if the URL cannot be parsed, credentials are missing,
/// the HTTP request fails, or the file cannot be read.
pub fn cloud_read_bytes(url: &str, config: &CloudConfig, http: &dyn HttpClient) -> Result<Vec<u8>> {
    let path = CloudPath::parse(url)?;
    match path {
        CloudPath::S3 { bucket, key, .. } => s3::s3_get_object(&bucket, &key, config, http),
        CloudPath::Gcs { bucket, key } => {
            // GCS has an S3-compatible XML API (storage.googleapis.com)
            // For now, return an informative error.
            Err(IoError::FormatError(format!(
                "GCS reading not yet implemented for gs://{bucket}/{key} — \
                 use S3-compatible interop or contribute a GCS backend"
            )))
        }
        CloudPath::Azure {
            account,
            container,
            blob,
        } => Err(IoError::FormatError(format!(
            "Azure Blob reading not yet implemented for {account}/{container}/{blob} — \
             contribute an Azure backend"
        ))),
        CloudPath::Local(fs_path) => {
            let data = std::fs::read(&fs_path)?;
            Ok(data)
        }
    }
}

// ───────────────────────────── Tests ─────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_cloud_path_parse_s3() {
        let path = CloudPath::parse("s3://bucket/key/file.csv").unwrap();
        assert_eq!(
            path,
            CloudPath::S3 {
                bucket: "bucket".to_string(),
                key: "key/file.csv".to_string(),
                region: None,
            }
        );
    }

    #[test]
    fn test_cloud_path_parse_gcs() {
        let path = CloudPath::parse("gs://bucket/key").unwrap();
        assert_eq!(
            path,
            CloudPath::Gcs {
                bucket: "bucket".to_string(),
                key: "key".to_string(),
            }
        );
    }

    #[test]
    fn test_cloud_path_parse_local() {
        let path = CloudPath::parse("/tmp/file.csv").unwrap();
        assert_eq!(path, CloudPath::Local(PathBuf::from("/tmp/file.csv")));
    }

    #[test]
    fn test_cloud_config_from_env() {
        let config = CloudConfig::from_env();
        // Just verify it doesn't panic — actual values depend on env
        let _ = format!("{config:?}");
    }

    #[test]
    fn test_cloud_read_bytes_local() {
        // Write a temp file and read it back
        let dir = std::env::temp_dir().join("scivex_cloud_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test_read.txt");
        {
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(b"hello cloud").unwrap();
        }

        let config = CloudConfig::default();
        let http = StdHttpClient;
        let data = cloud_read_bytes(path.to_str().unwrap(), &config, &http).unwrap();
        assert_eq!(data, b"hello cloud");

        // Cleanup
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn test_cloud_read_bytes_invalid_url() {
        let config = CloudConfig::default();
        let http = StdHttpClient;
        let result = cloud_read_bytes("", &config, &http);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_http_url_basic() {
        let (host, port, path) = parse_http_url("http://localhost:9000/bucket/key").unwrap();
        assert_eq!(host, "localhost");
        assert_eq!(port, 9000);
        assert_eq!(path, "/bucket/key");
    }

    #[test]
    fn test_parse_http_url_default_port() {
        let (host, port, path) = parse_http_url("http://example.com/foo").unwrap();
        assert_eq!(host, "example.com");
        assert_eq!(port, 80);
        assert_eq!(path, "/foo");
    }

    #[test]
    fn test_parse_http_url_https_rejected() {
        let result = parse_http_url("https://example.com/foo");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_status_code() {
        assert_eq!(parse_status_code("HTTP/1.1 200 OK").unwrap(), 200);
        assert_eq!(parse_status_code("HTTP/1.1 404 Not Found").unwrap(), 404);
    }

    #[test]
    fn test_find_header_end() {
        let data = b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nhello";
        let pos = find_header_end(data).unwrap();
        assert_eq!(&data[pos..], b"hello");
    }
}

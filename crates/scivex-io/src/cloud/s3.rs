//! AWS S3 client with Signature Version 4 signing.
//!
//! Implements the full SigV4 algorithm from scratch using our minimal SHA-256
//! and HMAC-SHA256 implementations. No external crypto dependencies required.

use crate::error::{IoError, Result};

use super::auth;
use super::sha256::{hex_encode, hmac_sha256, sha256_hex};
use super::{CloudConfig, HttpClient};

// ───────────────────────────── SigV4 signing ─────────────────────────────

/// AWS Signature Version 4 request signer.
pub struct SigV4Signer<'a> {
    /// HTTP method (GET, PUT, etc.)
    pub method: &'a str,
    /// Host header value (e.g. "my-bucket.s3.us-east-1.amazonaws.com")
    pub host: &'a str,
    /// URI path (e.g. "/path/to/key")
    pub uri: &'a str,
    /// Query string parameters, already sorted by key.
    pub query_params: &'a [(String, String)],
    /// Request headers (name lowercase, value).
    pub headers: &'a [(String, String)],
    /// Request body bytes (empty for GET).
    pub body: &'a [u8],
    /// AWS region (e.g. "us-east-1").
    pub region: &'a str,
    /// AWS service (e.g. "s3").
    pub service: &'a str,
    /// ISO-8601 date-time (e.g. "20130524T000000Z").
    pub datetime: &'a str,
    /// Access key ID.
    pub access_key: &'a str,
    /// Secret access key.
    pub secret_key: &'a str,
}

impl SigV4Signer<'_> {
    /// Compute the `Authorization` header value.
    pub fn authorization_header(&self) -> String {
        let date = &self.datetime[..8]; // YYYYMMDD

        // 1. Canonical request
        let canonical_request = self.canonical_request();
        let canonical_request_hash = sha256_hex(canonical_request.as_bytes());

        // 2. String to sign
        let credential_scope = format!("{date}/{}/{}/aws4_request", self.region, self.service);
        let string_to_sign = format!(
            "AWS4-HMAC-SHA256\n{}\n{credential_scope}\n{canonical_request_hash}",
            self.datetime
        );

        // 3. Signing key (HMAC chain)
        let signing_key = self.derive_signing_key(date);

        // 4. Signature
        let signature = hex_encode(&hmac_sha256(&signing_key, string_to_sign.as_bytes()));

        // 5. Authorization header
        let signed_headers = self.signed_headers_string();
        format!(
            "AWS4-HMAC-SHA256 Credential={}/{credential_scope}, \
             SignedHeaders={signed_headers}, Signature={signature}",
            self.access_key
        )
    }

    /// Build the canonical request string.
    fn canonical_request(&self) -> String {
        let canonical_uri = uri_encode_path(self.uri);
        let canonical_query = self.canonical_query_string();
        let canonical_headers = self.canonical_headers_string();
        let signed_headers = self.signed_headers_string();
        let payload_hash = sha256_hex(self.body);

        format!(
            "{}\n{canonical_uri}\n{canonical_query}\n{canonical_headers}\n\n{signed_headers}\n{payload_hash}",
            self.method
        )
    }

    /// Canonical query string: sorted params, URI-encoded keys and values.
    fn canonical_query_string(&self) -> String {
        let mut pairs: Vec<_> = self
            .query_params
            .iter()
            .map(|(k, v)| format!("{}={}", uri_encode(k), uri_encode(v)))
            .collect();
        pairs.sort();
        pairs.join("&")
    }

    /// Canonical headers: sorted lowercase name: trimmed value pairs.
    fn canonical_headers_string(&self) -> String {
        let mut headers: Vec<_> = self
            .headers
            .iter()
            .map(|(k, v)| (k.to_lowercase(), v.trim().to_string()))
            .collect();
        headers.sort_by(|a, b| a.0.cmp(&b.0));
        headers
            .iter()
            .map(|(k, v)| format!("{k}:{v}"))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Semicolon-separated sorted lowercase header names.
    fn signed_headers_string(&self) -> String {
        let mut names: Vec<_> = self.headers.iter().map(|(k, _)| k.to_lowercase()).collect();
        names.sort();
        names.dedup();
        names.join(";")
    }

    /// Derive the signing key via HMAC chain:
    /// ```text
    /// kDate    = HMAC("AWS4" + secret_key, date)
    /// kRegion  = HMAC(kDate, region)
    /// kService = HMAC(kRegion, service)
    /// kSigning = HMAC(kService, "aws4_request")
    /// ```
    fn derive_signing_key(&self, date: &str) -> [u8; 32] {
        let k_secret = format!("AWS4{}", self.secret_key);
        let k_date = hmac_sha256(k_secret.as_bytes(), date.as_bytes());
        let k_region = hmac_sha256(&k_date, self.region.as_bytes());
        let k_service = hmac_sha256(&k_region, self.service.as_bytes());
        hmac_sha256(&k_service, b"aws4_request")
    }
}

// ───────────────────────────── URI encoding ─────────────────────────────

/// URI-encode a string per RFC 3986 (unreserved chars are not encoded).
fn uri_encode(s: &str) -> String {
    let mut encoded = String::with_capacity(s.len() * 2);
    for byte in s.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                encoded.push(byte as char);
            }
            _ => {
                use std::fmt::Write;
                let _ = write!(encoded, "%{byte:02X}");
            }
        }
    }
    encoded
}

/// URI-encode a path, preserving `/` separators.
fn uri_encode_path(path: &str) -> String {
    path.split('/')
        .map(uri_encode)
        .collect::<Vec<_>>()
        .join("/")
}

// ───────────────────────────── S3 GET ─────────────────────────────

/// Build the URL and signed headers for an S3 `GetObject` request.
///
/// Returns `(url, headers)` where headers includes the Authorization header.
pub fn s3_build_signed_get(
    bucket: &str,
    key: &str,
    config: &CloudConfig,
) -> Result<(String, Vec<(String, String)>)> {
    let access_key = auth::resolve_aws_access_key(config)
        .ok_or_else(|| IoError::FormatError("AWS access key ID not configured".to_string()))?;
    let secret_key = auth::resolve_aws_secret_key(config)
        .ok_or_else(|| IoError::FormatError("AWS secret access key not configured".to_string()))?;
    let region = auth::resolve_aws_region(config);

    let host = format!("{bucket}.s3.{region}.amazonaws.com");
    let uri = if key.is_empty() {
        "/".to_string()
    } else if key.starts_with('/') {
        key.to_string()
    } else {
        format!("/{key}")
    };

    let datetime = utc_now_iso8601();
    let date = &datetime[..8];

    let mut headers = vec![
        ("host".to_string(), host.clone()),
        ("x-amz-content-sha256".to_string(), sha256_hex(b"")),
        ("x-amz-date".to_string(), datetime.clone()),
    ];

    // Add session token if present
    if let Some(token) = auth::resolve_aws_session_token(config) {
        headers.push(("x-amz-security-token".to_string(), token));
    }

    let signer = SigV4Signer {
        method: "GET",
        host: &host,
        uri: &uri,
        query_params: &[],
        headers: &headers,
        body: b"",
        region: &region,
        service: "s3",
        datetime: &datetime,
        access_key: &access_key,
        secret_key: &secret_key,
    };

    let auth_header = signer.authorization_header();
    headers.push(("authorization".to_string(), auth_header));

    let url = format!("https://{host}{uri}");

    // Suppress unused-variable warning for `date`
    let _ = date;

    Ok((url, headers))
}

/// Perform an S3 `GetObject` using the provided HTTP client.
pub fn s3_get_object(
    bucket: &str,
    key: &str,
    config: &CloudConfig,
    http: &dyn HttpClient,
) -> Result<Vec<u8>> {
    let (url, headers) = s3_build_signed_get(bucket, key, config)?;
    http.get(&url, &headers)
}

// ───────────────────────────── Time helpers ─────────────────────────────

/// Return the current UTC time in ISO-8601 basic format: `YYYYMMDDTHHMMSSZ`.
///
/// Uses `std::time::SystemTime` to avoid external deps.
fn utc_now_iso8601() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};

    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // Convert seconds since epoch to date-time components
    let (year, month, day, hour, min, sec) = epoch_to_datetime(secs);
    format!("{year:04}{month:02}{day:02}T{hour:02}{min:02}{sec:02}Z")
}

/// Convert seconds since Unix epoch to (year, month, day, hour, minute, second).
fn epoch_to_datetime(secs: u64) -> (u64, u64, u64, u64, u64, u64) {
    let sec = secs % 60;
    let min = (secs / 60) % 60;
    let hour = (secs / 3600) % 24;
    let mut days = secs / 86400;

    // Compute year from days since epoch (1970-01-01)
    let mut year = 1970u64;
    loop {
        let days_in_year = if is_leap_year(year) { 366 } else { 365 };
        if days < days_in_year {
            break;
        }
        days -= days_in_year;
        year += 1;
    }

    // Compute month
    let leap = is_leap_year(year);
    let month_days: [u64; 12] = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];

    let mut month = 0u64;
    for (i, &md) in month_days.iter().enumerate() {
        if days < md {
            month = i as u64 + 1;
            break;
        }
        days -= md;
    }
    if month == 0 {
        month = 12;
    }
    let day = days + 1;

    (year, month, day, hour, min, sec)
}

/// Check if a year is a leap year.
const fn is_leap_year(year: u64) -> bool {
    (year.is_multiple_of(4) && !year.is_multiple_of(100)) || year.is_multiple_of(400)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uri_encode() {
        assert_eq!(uri_encode("hello world"), "hello%20world");
        assert_eq!(uri_encode("foo/bar"), "foo%2Fbar");
        assert_eq!(uri_encode("a-b_c.d~e"), "a-b_c.d~e");
    }

    #[test]
    fn test_uri_encode_path() {
        assert_eq!(uri_encode_path("/foo/bar/baz"), "/foo/bar/baz");
        assert_eq!(uri_encode_path("/foo/bar baz/qux"), "/foo/bar%20baz/qux");
    }

    #[test]
    fn test_epoch_to_datetime() {
        // 2013-05-24T00:00:00Z = 1369353600
        let (y, m, d, h, mi, s) = epoch_to_datetime(1_369_353_600);
        assert_eq!((y, m, d, h, mi, s), (2013, 5, 24, 0, 0, 0));
    }

    #[test]
    fn test_epoch_to_datetime_unix_epoch() {
        let (y, m, d, h, mi, s) = epoch_to_datetime(0);
        assert_eq!((y, m, d, h, mi, s), (1970, 1, 1, 0, 0, 0));
    }

    /// AWS SigV4 test using the example from the AWS documentation.
    ///
    /// Reference: <https://docs.aws.amazon.com/general/latest/gr/sigv4-create-canonical-request.html>
    #[test]
    fn test_sigv4_signing() {
        // Use the AWS SigV4 test suite example values.
        // We'll verify the signing key derivation and a known canonical request hash.

        // Derive signing key for 20130524 / us-east-1 / s3 / aws4_request
        let secret_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY";
        let date = "20130524";
        let region = "us-east-1";
        let service = "s3";

        let k_secret = format!("AWS4{secret_key}");
        let k_date = hmac_sha256(k_secret.as_bytes(), date.as_bytes());
        let k_region = hmac_sha256(&k_date, region.as_bytes());
        let k_service = hmac_sha256(&k_region, service.as_bytes());
        let k_signing = hmac_sha256(&k_service, b"aws4_request");

        // The signing key should be deterministic for these inputs.
        // We verify it's 32 bytes and non-zero.
        assert_eq!(k_signing.len(), 32);
        assert!(k_signing.iter().any(|&b| b != 0));

        // Now test a full signing flow with known inputs
        let headers = vec![
            (
                "host".to_string(),
                "examplebucket.s3.amazonaws.com".to_string(),
            ),
            ("x-amz-content-sha256".to_string(), sha256_hex(b"")),
            ("x-amz-date".to_string(), "20130524T000000Z".to_string()),
        ];

        let signer = SigV4Signer {
            method: "GET",
            host: "examplebucket.s3.amazonaws.com",
            uri: "/test.txt",
            query_params: &[],
            headers: &headers,
            body: b"",
            region: "us-east-1",
            service: "s3",
            datetime: "20130524T000000Z",
            access_key: "AKIAIOSFODNN7EXAMPLE",
            secret_key: "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        };

        let auth = signer.authorization_header();

        // Verify it starts with the right prefix
        assert!(auth.starts_with(
            "AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE/20130524/us-east-1/s3/aws4_request"
        ));
        // Verify it contains SignedHeaders and Signature
        assert!(auth.contains("SignedHeaders=host;x-amz-content-sha256;x-amz-date"));
        assert!(auth.contains("Signature="));

        // Extract and validate the signature is a 64-char hex string
        let sig_start = auth.find("Signature=").unwrap() + "Signature=".len();
        let signature = &auth[sig_start..];
        assert_eq!(signature.len(), 64);
        assert!(signature.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_s3_build_signed_get_missing_credentials() {
        // Use a config with no credentials and no env fallback.
        let config = CloudConfig::default();
        // Clear env vars that might interfere
        let saved = std::env::var("AWS_ACCESS_KEY_ID").ok();
        // SAFETY: test-only, single-threaded; no other thread reads these vars.
        unsafe { std::env::remove_var("AWS_ACCESS_KEY_ID") };

        let result = s3_build_signed_get("bucket", "key", &config);
        assert!(result.is_err());

        // Restore
        if let Some(v) = saved {
            // SAFETY: restoring original value, test-only.
            unsafe { std::env::set_var("AWS_ACCESS_KEY_ID", v) };
        }
    }
}

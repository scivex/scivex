//! Cloud storage URL parsing.
//!
//! Recognises the following schemes:
//!
//! - `s3://bucket/key` — Amazon S3
//! - `gs://bucket/key` — Google Cloud Storage
//! - `az://container/blob` — Azure Blob Storage (simple form)
//! - `wasbs://container@account.blob.core.windows.net/path` — Azure (WASBS form)
//! - `file:///path` or plain filesystem paths — Local

use std::path::PathBuf;

use crate::error::{IoError, Result};

use super::CloudPath;

/// Parse a URL string into a [`CloudPath`].
pub fn parse_cloud_url(url: &str) -> Result<CloudPath> {
    let trimmed = url.trim();
    if trimmed.is_empty() {
        return Err(IoError::FormatError("empty URL".to_string()));
    }

    // S3
    if let Some(rest) = trimmed.strip_prefix("s3://") {
        return parse_s3(rest);
    }

    // GCS
    if let Some(rest) = trimmed.strip_prefix("gs://") {
        return parse_gcs(rest);
    }

    // Azure simple form
    if let Some(rest) = trimmed.strip_prefix("az://") {
        return parse_azure_simple(rest);
    }

    // Azure WASBS form
    if let Some(rest) = trimmed.strip_prefix("wasbs://") {
        return parse_azure_wasbs(rest);
    }

    // file:// scheme
    if let Some(rest) = trimmed.strip_prefix("file://") {
        return Ok(CloudPath::Local(PathBuf::from(rest)));
    }

    // No scheme → treat as local path
    Ok(CloudPath::Local(PathBuf::from(trimmed)))
}

/// Parse `bucket/key` from the remainder after `s3://`.
fn parse_s3(rest: &str) -> Result<CloudPath> {
    if rest.is_empty() {
        return Err(IoError::FormatError(
            "S3 URL missing bucket name".to_string(),
        ));
    }
    let (bucket, key) = split_bucket_key(rest)?;
    Ok(CloudPath::S3 {
        bucket,
        key,
        region: None,
    })
}

/// Parse `bucket/key` from the remainder after `gs://`.
fn parse_gcs(rest: &str) -> Result<CloudPath> {
    if rest.is_empty() {
        return Err(IoError::FormatError(
            "GCS URL missing bucket name".to_string(),
        ));
    }
    let (bucket, key) = split_bucket_key(rest)?;
    Ok(CloudPath::Gcs { bucket, key })
}

/// Parse `container/blob` from the remainder after `az://`.
fn parse_azure_simple(rest: &str) -> Result<CloudPath> {
    if rest.is_empty() {
        return Err(IoError::FormatError(
            "Azure URL missing container name".to_string(),
        ));
    }
    let (container, blob) = split_bucket_key(rest)?;
    // For simple form, we don't have account info — leave it empty
    // The caller must supply the account via `CloudConfig`.
    Ok(CloudPath::Azure {
        account: String::new(),
        container,
        blob,
    })
}

/// Parse `container@account.blob.core.windows.net/path` from the remainder
/// after `wasbs://`.
fn parse_azure_wasbs(rest: &str) -> Result<CloudPath> {
    // Format: container@account.blob.core.windows.net/path
    let at_pos = rest
        .find('@')
        .ok_or_else(|| IoError::FormatError("WASBS URL missing '@' separator".to_string()))?;
    let container = rest[..at_pos].to_string();
    let after_at = &rest[at_pos + 1..];

    // Extract account from "account.blob.core.windows.net/path"
    let dot_pos = after_at
        .find('.')
        .ok_or_else(|| IoError::FormatError("WASBS URL missing account host".to_string()))?;
    let account = after_at[..dot_pos].to_string();

    // Find the path after the host
    let blob = if let Some(slash_pos) = after_at.find('/') {
        after_at[slash_pos + 1..].to_string()
    } else {
        String::new()
    };

    if container.is_empty() {
        return Err(IoError::FormatError(
            "WASBS URL has empty container".to_string(),
        ));
    }
    if account.is_empty() {
        return Err(IoError::FormatError(
            "WASBS URL has empty account".to_string(),
        ));
    }

    Ok(CloudPath::Azure {
        account,
        container,
        blob,
    })
}

/// Split `"bucket/path/to/key"` into `("bucket", "path/to/key")`.
fn split_bucket_key(s: &str) -> Result<(String, String)> {
    match s.find('/') {
        Some(pos) => {
            let bucket = s[..pos].to_string();
            let key = s[pos + 1..].to_string();
            if bucket.is_empty() {
                return Err(IoError::FormatError(
                    "URL has empty bucket/container name".to_string(),
                ));
            }
            Ok((bucket, key))
        }
        None => Ok((s.to_string(), String::new())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_s3() {
        let path = parse_cloud_url("s3://my-bucket/data/file.csv").unwrap();
        assert_eq!(
            path,
            CloudPath::S3 {
                bucket: "my-bucket".to_string(),
                key: "data/file.csv".to_string(),
                region: None,
            }
        );
    }

    #[test]
    fn test_parse_s3_no_key() {
        let path = parse_cloud_url("s3://my-bucket").unwrap();
        assert_eq!(
            path,
            CloudPath::S3 {
                bucket: "my-bucket".to_string(),
                key: String::new(),
                region: None,
            }
        );
    }

    #[test]
    fn test_parse_gcs() {
        let path = parse_cloud_url("gs://my-bucket/data/file.parquet").unwrap();
        assert_eq!(
            path,
            CloudPath::Gcs {
                bucket: "my-bucket".to_string(),
                key: "data/file.parquet".to_string(),
            }
        );
    }

    #[test]
    fn test_parse_azure_simple() {
        let path = parse_cloud_url("az://mycontainer/path/to/blob").unwrap();
        assert_eq!(
            path,
            CloudPath::Azure {
                account: String::new(),
                container: "mycontainer".to_string(),
                blob: "path/to/blob".to_string(),
            }
        );
    }

    #[test]
    fn test_parse_azure_wasbs() {
        let path =
            parse_cloud_url("wasbs://mycontainer@myaccount.blob.core.windows.net/path/to/blob")
                .unwrap();
        assert_eq!(
            path,
            CloudPath::Azure {
                account: "myaccount".to_string(),
                container: "mycontainer".to_string(),
                blob: "path/to/blob".to_string(),
            }
        );
    }

    #[test]
    fn test_parse_file_scheme() {
        let path = parse_cloud_url("file:///tmp/data.csv").unwrap();
        assert_eq!(path, CloudPath::Local(PathBuf::from("/tmp/data.csv")));
    }

    #[test]
    fn test_parse_local_path() {
        let path = parse_cloud_url("/home/user/data.csv").unwrap();
        assert_eq!(path, CloudPath::Local(PathBuf::from("/home/user/data.csv")));
    }

    #[test]
    fn test_parse_relative_local_path() {
        let path = parse_cloud_url("data/file.csv").unwrap();
        assert_eq!(path, CloudPath::Local(PathBuf::from("data/file.csv")));
    }

    #[test]
    fn test_parse_empty_url() {
        assert!(parse_cloud_url("").is_err());
    }

    #[test]
    fn test_parse_s3_empty_bucket() {
        assert!(parse_cloud_url("s3://").is_err());
    }
}

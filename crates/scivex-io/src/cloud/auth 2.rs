//! Cloud credential resolution.
//!
//! Credentials are resolved in order:
//! 1. Explicit fields in [`CloudConfig`]
//! 2. Environment variables

use super::CloudConfig;

/// Resolve AWS access key ID from config or environment.
pub fn resolve_aws_access_key(config: &CloudConfig) -> Option<String> {
    config
        .aws_access_key_id
        .clone()
        .or_else(|| std::env::var("AWS_ACCESS_KEY_ID").ok())
}

/// Resolve AWS secret access key from config or environment.
pub fn resolve_aws_secret_key(config: &CloudConfig) -> Option<String> {
    config
        .aws_secret_access_key
        .clone()
        .or_else(|| std::env::var("AWS_SECRET_ACCESS_KEY").ok())
}

/// Resolve AWS region from config or environment.
pub fn resolve_aws_region(config: &CloudConfig) -> String {
    config
        .aws_region
        .clone()
        .or_else(|| std::env::var("AWS_DEFAULT_REGION").ok())
        .or_else(|| std::env::var("AWS_REGION").ok())
        .unwrap_or_else(|| "us-east-1".to_string())
}

/// Resolve AWS session token from config or environment.
pub fn resolve_aws_session_token(config: &CloudConfig) -> Option<String> {
    config
        .aws_session_token
        .clone()
        .or_else(|| std::env::var("AWS_SESSION_TOKEN").ok())
}

/// Resolve GCS service account key path from config or environment.
pub fn resolve_gcs_service_account_key(config: &CloudConfig) -> Option<String> {
    config
        .gcs_service_account_key
        .clone()
        .or_else(|| std::env::var("GOOGLE_APPLICATION_CREDENTIALS").ok())
}

/// Resolve Azure storage account from config or environment.
pub fn resolve_azure_account(config: &CloudConfig) -> Option<String> {
    config
        .azure_storage_account
        .clone()
        .or_else(|| std::env::var("AZURE_STORAGE_ACCOUNT").ok())
}

/// Resolve Azure storage key from config or environment.
pub fn resolve_azure_storage_key(config: &CloudConfig) -> Option<String> {
    config
        .azure_storage_key
        .clone()
        .or_else(|| std::env::var("AZURE_STORAGE_KEY").ok())
}

/// Resolve Azure SAS token from config or environment.
pub fn resolve_azure_sas_token(config: &CloudConfig) -> Option<String> {
    config
        .azure_sas_token
        .clone()
        .or_else(|| std::env::var("AZURE_SAS_TOKEN").ok())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Serialize env-var tests to avoid races.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    // SAFETY for all set_var/remove_var calls in these tests:
    // We hold ENV_LOCK, ensuring no other test thread is reading or writing
    // environment variables concurrently. Tests in this module are the only
    // callers that mutate these specific env vars.

    unsafe fn set_env(key: &str, val: &str) {
        // SAFETY: caller holds ENV_LOCK, no concurrent env mutation.
        unsafe { std::env::set_var(key, val) }
    }

    unsafe fn remove_env(key: &str) {
        // SAFETY: caller holds ENV_LOCK, no concurrent env mutation.
        unsafe { std::env::remove_var(key) }
    }

    unsafe fn restore_env(key: &str, saved: Option<String>) {
        // SAFETY: caller holds ENV_LOCK, no concurrent env mutation.
        match saved {
            Some(v) => unsafe { std::env::set_var(key, v) },
            None => unsafe { std::env::remove_var(key) },
        }
    }

    #[test]
    fn test_config_fields_take_priority() {
        let config = CloudConfig {
            aws_access_key_id: Some("from-config".to_string()),
            aws_secret_access_key: Some("secret-from-config".to_string()),
            aws_region: Some("eu-west-1".to_string()),
            ..CloudConfig::default()
        };
        assert_eq!(
            resolve_aws_access_key(&config),
            Some("from-config".to_string())
        );
        assert_eq!(
            resolve_aws_secret_key(&config),
            Some("secret-from-config".to_string())
        );
        assert_eq!(resolve_aws_region(&config), "eu-west-1");
    }

    #[test]
    fn test_env_fallback() {
        let _guard = ENV_LOCK.lock().unwrap();
        let saved_key = std::env::var("AWS_ACCESS_KEY_ID").ok();
        let saved_secret = std::env::var("AWS_SECRET_ACCESS_KEY").ok();
        let saved_region = std::env::var("AWS_DEFAULT_REGION").ok();

        // SAFETY: ENV_LOCK is held; no concurrent env mutation.
        unsafe {
            set_env("AWS_ACCESS_KEY_ID", "env-key");
            set_env("AWS_SECRET_ACCESS_KEY", "env-secret");
            set_env("AWS_DEFAULT_REGION", "ap-southeast-1");
        }

        let config = CloudConfig::default();
        assert_eq!(resolve_aws_access_key(&config), Some("env-key".to_string()));
        assert_eq!(
            resolve_aws_secret_key(&config),
            Some("env-secret".to_string())
        );
        assert_eq!(resolve_aws_region(&config), "ap-southeast-1");

        // SAFETY: ENV_LOCK is held; no concurrent env mutation.
        unsafe {
            restore_env("AWS_ACCESS_KEY_ID", saved_key);
            restore_env("AWS_SECRET_ACCESS_KEY", saved_secret);
            restore_env("AWS_DEFAULT_REGION", saved_region);
        }
    }

    #[test]
    fn test_default_region() {
        let _guard = ENV_LOCK.lock().unwrap();
        let saved_default = std::env::var("AWS_DEFAULT_REGION").ok();
        let saved_region = std::env::var("AWS_REGION").ok();

        // SAFETY: ENV_LOCK is held; no concurrent env mutation.
        unsafe {
            remove_env("AWS_DEFAULT_REGION");
            remove_env("AWS_REGION");
        }

        let config = CloudConfig::default();
        assert_eq!(resolve_aws_region(&config), "us-east-1");

        // SAFETY: ENV_LOCK is held; no concurrent env mutation.
        unsafe {
            restore_env("AWS_DEFAULT_REGION", saved_default);
            restore_env("AWS_REGION", saved_region);
        }
    }
}

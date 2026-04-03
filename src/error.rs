/// All fallible femind operations return `Result<T>`.
pub type Result<T> = std::result::Result<T, FemindError>;

/// Unified error type for all femind operations.
///
/// Consumers can match on variants to handle specific failure domains
/// (e.g., retry on transient `Database` errors, surface `ModelNotAvailable`
/// to users, fall back to FTS5 on `ModelMismatch`).
#[derive(Debug, thiserror::Error)]
pub enum FemindError {
    /// SQLite database error (connection, query, constraint violation).
    #[error("database error: {0}")]
    Database(#[from] rusqlite::Error),

    /// Embedding inference failed (model load, tensor operation, tokenization).
    #[error("embedding error: {0}")]
    Embedding(String),

    /// Requested embedding model is not available (download failed, not bundled).
    #[error("model not available: {0}")]
    ModelNotAvailable(String),

    /// Remote service rejected the request because credentials were missing or invalid.
    #[error("remote auth error: {0}")]
    RemoteAuth(String),

    /// Remote service timed out.
    #[error("remote timeout: {0}")]
    RemoteTimeout(String),

    /// Remote service transport failed before a valid response was received.
    #[error("remote transport error: {0}")]
    RemoteTransport(String),

    /// Remote service was unavailable or returned a temporary failure.
    #[error("remote unavailable: {0}")]
    RemoteUnavailable(String),

    /// JSON serialization/deserialization of MemoryRecord or metadata failed.
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Stored vectors were produced by a different model than the current backend.
    ///
    /// In normal search flow this does not surface — the engine silently falls
    /// back to FTS5. Only raised when the caller explicitly requests
    /// `SearchMode::Vector` and no compatible vectors exist.
    #[error("model mismatch: stored with '{stored}', current backend is '{current}'")]
    ModelMismatch {
        /// The model name stored alongside the vectors.
        stored: String,
        /// The current embedding backend's model name.
        current: String,
    },

    /// Remote service reported a profile or model mismatch.
    #[error("remote profile mismatch: {0}")]
    RemoteProfileMismatch(String),

    /// Schema migration failed (version mismatch, DDL error).
    #[error("migration error: {0}")]
    Migration(String),

    /// Encryption operation failed (wrong key, SQLCipher error).
    #[cfg(feature = "encryption")]
    #[error("encryption error: {0}")]
    Encryption(String),

    /// Consolidation pipeline error (strategy failure, conflict resolution).
    #[error("consolidation error: {0}")]
    Consolidation(String),

    /// LLM callback returned an error.
    #[error("llm callback error: {0}")]
    LlmCallback(String),

    /// I/O error (file access, model download).
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display() {
        let err = FemindError::Embedding("tensor shape mismatch".into());
        assert_eq!(err.to_string(), "embedding error: tensor shape mismatch");
    }

    #[test]
    fn model_mismatch_display() {
        let err = FemindError::ModelMismatch {
            stored: "model-a".into(),
            current: "model-b".into(),
        };
        assert!(err.to_string().contains("model-a"));
        assert!(err.to_string().contains("model-b"));
    }

    #[test]
    fn remote_error_display() {
        let auth = FemindError::RemoteAuth("token missing".into());
        let mismatch = FemindError::RemoteProfileMismatch("expected foo".into());
        assert!(auth.to_string().contains("remote auth error"));
        assert!(mismatch.to_string().contains("remote profile mismatch"));
    }

    #[test]
    fn from_rusqlite_error() {
        let sqlite_err = rusqlite::Error::QueryReturnedNoRows;
        let err: FemindError = sqlite_err.into();
        matches!(err, FemindError::Database(_));
    }

    #[test]
    fn result_type_alias() {
        fn returns_ok() -> Result<i32> {
            Ok(42)
        }
        fn returns_err() -> Result<i32> {
            Err(FemindError::Migration("test".into()))
        }
        assert!(returns_ok().is_ok());
        assert!(returns_err().is_err());
    }
}

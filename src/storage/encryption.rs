/// Encryption key types for SQLCipher database encryption.
///
/// Only available when the `encryption` feature is enabled.
/// The key must be provided before any database operations.
#[derive(Debug, Clone)]
pub enum EncryptionKey {
    /// Raw passphrase — SQLCipher derives the key via PBKDF2 (256K iterations).
    Passphrase(String),
    /// Pre-derived raw key bytes (256-bit AES key).
    RawKey([u8; 32]),
}

impl EncryptionKey {
    /// Format the key as a PRAGMA statement value.
    pub fn as_pragma_value(&self) -> String {
        match self {
            Self::Passphrase(pass) => format!("'{}'", pass.replace('\'', "''")),
            Self::RawKey(bytes) => {
                let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
                format!("\"x'{hex}'\"")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passphrase_pragma() {
        let key = EncryptionKey::Passphrase("my secret".into());
        let pragma = key.as_pragma_value();
        assert!(pragma.contains("my secret"));
    }

    #[test]
    fn raw_key_pragma() {
        let key = EncryptionKey::RawKey([0xab; 32]);
        let pragma = key.as_pragma_value();
        assert!(pragma.contains("abab"));
    }

    #[test]
    fn passphrase_escapes_quotes() {
        let key = EncryptionKey::Passphrase("it's a test".into());
        let pragma = key.as_pragma_value();
        assert!(pragma.contains("it''s a test"));
    }
}

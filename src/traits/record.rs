use chrono::{DateTime, Utc};
use serde::{de::DeserializeOwned, Serialize};
use std::collections::HashMap;

/// Cognitive memory type classification (CoALA framework).
///
/// Determines decay rate, scoring behavior, and pruning eligibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, serde::Deserialize)]
pub enum MemoryType {
    /// What happened — events, sessions, iteration logs.
    /// Decays fastest. Most useful when recent.
    Episodic,
    /// What I know — facts, preferences, project context.
    /// Stable over time. Core knowledge.
    Semantic,
    /// How to do things — workflows, error patterns, solutions.
    /// Strengthens with validation. Most valuable when proven.
    Procedural,
}

impl MemoryType {
    /// Convert to the string stored in SQLite.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Episodic => "episodic",
            Self::Semantic => "semantic",
            Self::Procedural => "procedural",
        }
    }

    /// Parse from the string stored in SQLite.
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "episodic" => Some(Self::Episodic),
            "semantic" => Some(Self::Semantic),
            "procedural" => Some(Self::Procedural),
            _ => None,
        }
    }
}

impl std::fmt::Display for MemoryType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Consumers implement this for their memory types.
///
/// femind handles storage, indexing, search, and decay.
/// The consumer defines what a "memory" is by implementing this trait.
///
/// # Object Safety
///
/// This trait is NOT object-safe due to `Serialize + DeserializeOwned` bounds.
/// For scoring, consolidation, and evolution strategies that need runtime dispatch,
/// use [`MemoryMeta`] instead — the engine extracts it from `T: MemoryRecord`
/// before passing to those trait objects.
pub trait MemoryRecord: Send + Sync + Serialize + DeserializeOwned + 'static {
    /// Unique identifier. `None` for records not yet stored.
    fn id(&self) -> Option<i64>;

    /// Text to embed (vector search) and index (FTS5).
    fn searchable_text(&self) -> String;

    /// Cognitive memory type — determines decay rate and scoring.
    fn memory_type(&self) -> MemoryType;

    /// Importance score (1-10, default 5). Affects scoring.
    fn importance(&self) -> u8 {
        5
    }

    /// When this memory was created.
    fn created_at(&self) -> DateTime<Utc>;

    /// Optional category for boost matching (e.g., "error", "decision", "pattern").
    fn category(&self) -> Option<&str> {
        None
    }

    /// Optional metadata for filtering (key-value pairs).
    fn metadata(&self) -> HashMap<String, String> {
        HashMap::new()
    }

    /// Optional temporal validity — when this fact became true.
    #[cfg(feature = "temporal")]
    fn valid_from(&self) -> Option<DateTime<Utc>> {
        None
    }

    /// Optional temporal validity — when this fact stopped being true.
    #[cfg(feature = "temporal")]
    fn valid_until(&self) -> Option<DateTime<Utc>> {
        None
    }
}

/// Extracted metadata from a `MemoryRecord` for use in scoring, consolidation,
/// and evolution traits.
///
/// These traits cannot use `dyn MemoryRecord` because `MemoryRecord` requires
/// `Serialize + DeserializeOwned` (not object-safe). The engine extracts
/// `MemoryMeta` from `T: MemoryRecord` before passing to strategy trait objects.
#[derive(Debug, Clone)]
pub struct MemoryMeta {
    /// Database row ID (`None` if not yet stored).
    pub id: Option<i64>,
    /// The searchable text content.
    pub searchable_text: String,
    /// Cognitive memory type.
    pub memory_type: MemoryType,
    /// Importance score (1-10).
    pub importance: u8,
    /// Optional category.
    pub category: Option<String>,
    /// Creation timestamp.
    pub created_at: DateTime<Utc>,
    /// Key-value metadata.
    pub metadata: HashMap<String, String>,
}

impl MemoryMeta {
    /// Extract metadata from a `MemoryRecord`.
    pub fn from_record<T: MemoryRecord>(record: &T) -> Self {
        Self {
            id: record.id(),
            searchable_text: record.searchable_text(),
            memory_type: record.memory_type(),
            importance: record.importance(),
            category: record.category().map(String::from),
            created_at: record.created_at(),
            metadata: record.metadata(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, Serialize, serde::Deserialize)]
    struct TestMemory {
        id: Option<i64>,
        text: String,
        created_at: DateTime<Utc>,
    }

    impl MemoryRecord for TestMemory {
        fn id(&self) -> Option<i64> {
            self.id
        }
        fn searchable_text(&self) -> String {
            self.text.clone()
        }
        fn memory_type(&self) -> MemoryType {
            MemoryType::Semantic
        }
        fn created_at(&self) -> DateTime<Utc> {
            self.created_at
        }
    }

    #[test]
    fn memory_type_roundtrip() {
        for mt in [
            MemoryType::Episodic,
            MemoryType::Semantic,
            MemoryType::Procedural,
        ] {
            let s = mt.as_str();
            let parsed = MemoryType::from_str(s);
            assert_eq!(parsed, Some(mt), "roundtrip failed for {s}");
        }
    }

    #[test]
    fn memory_type_display() {
        assert_eq!(MemoryType::Episodic.to_string(), "episodic");
        assert_eq!(MemoryType::Semantic.to_string(), "semantic");
        assert_eq!(MemoryType::Procedural.to_string(), "procedural");
    }

    #[test]
    fn memory_meta_from_record() {
        let record = TestMemory {
            id: Some(42),
            text: "test memory".to_string(),
            created_at: Utc::now(),
        };
        let meta = MemoryMeta::from_record(&record);
        assert_eq!(meta.id, Some(42));
        assert_eq!(meta.searchable_text, "test memory");
        assert_eq!(meta.memory_type, MemoryType::Semantic);
        assert_eq!(meta.importance, 5); // default
        assert!(meta.category.is_none());
    }

    #[test]
    fn memory_type_from_invalid() {
        assert_eq!(MemoryType::from_str("invalid"), None);
        assert_eq!(MemoryType::from_str(""), None);
    }
}

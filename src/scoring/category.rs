use crate::traits::{MemoryMeta, ScoringStrategy};

/// Boost score when the memory's category matches keywords in the query.
///
/// For example, if the query contains "error" or "failed" and the memory
/// has category "error", it gets a boost.
pub struct CategoryScorer {
    boost: f32,
}

impl CategoryScorer {
    /// Create with a custom boost multiplier (applied when category matches).
    pub fn new(boost: f32) -> Self {
        Self { boost }
    }
}

impl Default for CategoryScorer {
    fn default() -> Self {
        Self::new(1.3)
    }
}

impl ScoringStrategy for CategoryScorer {
    fn score_multiplier(&self, record: &MemoryMeta, query: &str, _base_score: f32) -> f32 {
        let Some(ref category) = record.category else {
            return 1.0;
        };

        let query_lower = query.to_lowercase();
        let cat_lower = category.to_lowercase();

        // Direct match: query contains the category name
        if query_lower.contains(&cat_lower) {
            return self.boost;
        }

        // Keyword hints that suggest a category
        let category_hints: &[(&str, &[&str])] = &[
            ("error", &["error", "fail", "crash", "bug", "broken", "exception"]),
            ("decision", &["decided", "decision", "chose", "approach", "strategy"]),
            ("pattern", &["pattern", "workflow", "process", "how to"]),
            ("lesson", &["learned", "lesson", "insight", "takeaway"]),
        ];

        for (cat, hints) in category_hints {
            if cat_lower == *cat {
                for hint in *hints {
                    if query_lower.contains(hint) {
                        return self.boost;
                    }
                }
            }
        }

        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::MemoryType;
    use chrono::Utc;

    fn meta_cat(cat: &str) -> MemoryMeta {
        MemoryMeta {
            id: Some(1),
            searchable_text: "test".into(),
            memory_type: MemoryType::Semantic,
            importance: 5,
            category: Some(cat.into()),
            created_at: Utc::now(),
            metadata: Default::default(),
        }
    }

    #[test]
    fn direct_category_match() {
        let scorer = CategoryScorer::default();
        let m = scorer.score_multiplier(&meta_cat("error"), "show me the error", 1.0);
        assert!((m - 1.3).abs() < 0.01);
    }

    #[test]
    fn keyword_hint_match() {
        let scorer = CategoryScorer::default();
        let m = scorer.score_multiplier(&meta_cat("error"), "why did it fail", 1.0);
        assert!((m - 1.3).abs() < 0.01, "fail should hint at error category, got {m}");
    }

    #[test]
    fn no_match() {
        let scorer = CategoryScorer::default();
        let m = scorer.score_multiplier(&meta_cat("error"), "what is the architecture", 1.0);
        assert!((m - 1.0).abs() < 0.01);
    }

    #[test]
    fn no_category() {
        let scorer = CategoryScorer::default();
        let meta = MemoryMeta {
            category: None,
            ..meta_cat("error")
        };
        let m = scorer.score_multiplier(&meta, "error query", 1.0);
        assert!((m - 1.0).abs() < 0.01);
    }
}

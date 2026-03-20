use crate::traits::MemoryType;

/// Priority levels — lower number = included first in the context budget.
pub const PRIORITY_BEHAVIORAL: u8 = 0;
pub const PRIORITY_RETRY: u8 = 10;
pub const PRIORITY_SPEC: u8 = 15;
pub const PRIORITY_SIMILAR: u8 = 25;
pub const PRIORITY_LEARNING: u8 = 40;
pub const PRIORITY_HISTORICAL: u8 = 60;

/// Token budget configuration for context assembly.
#[derive(Debug, Clone)]
pub struct ContextBudget {
    /// Maximum tokens to spend on assembled context.
    pub max_tokens: usize,
    /// Approximate tokens per character (default: 0.25 for English text).
    pub tokens_per_char: f32,
}

impl ContextBudget {
    /// Create with a token limit and default character ratio.
    pub fn new(max_tokens: usize) -> Self {
        Self {
            max_tokens,
            tokens_per_char: 0.25,
        }
    }

    /// Create with a custom tokens-per-character ratio.
    pub fn with_ratio(max_tokens: usize, tokens_per_char: f32) -> Self {
        Self {
            max_tokens,
            tokens_per_char,
        }
    }

    /// Estimate tokens for a given text.
    pub fn estimate_tokens(&self, text: &str) -> usize {
        (text.len() as f32 * self.tokens_per_char).ceil() as usize
    }
}

impl Default for ContextBudget {
    fn default() -> Self {
        Self::new(4096)
    }
}

/// A single item that can be included in context assembly.
#[derive(Debug, Clone)]
pub struct ContextItem {
    /// Memory row ID.
    pub memory_id: i64,
    /// The text content to include.
    pub content: String,
    /// Priority level (lower = included first).
    pub priority: u8,
    /// Estimated token count.
    pub estimated_tokens: usize,
    /// Relevance score from search (for ordering within a priority level).
    pub relevance_score: f32,
    /// Memory type (for section grouping).
    pub memory_type: MemoryType,
    /// Optional category label.
    pub category: Option<String>,
}

/// The assembled context ready for LLM prompt injection.
#[derive(Debug, Clone)]
pub struct ContextAssembly {
    /// Ordered items included within the budget.
    pub items: Vec<ContextItem>,
    /// Total estimated tokens used.
    pub total_tokens: usize,
    /// Number of candidate items that were excluded (budget exceeded).
    pub excluded_count: usize,
    /// The maximum budget that was available.
    pub budget_max: usize,
}

impl ContextAssembly {
    /// Assemble context from candidate items within the given budget.
    ///
    /// Algorithm:
    /// 1. Sort by priority (ascending), then by relevance score (descending)
    /// 2. Add items until token budget is exhausted
    /// 3. Skip items that would exceed remaining budget
    pub fn assemble(mut candidates: Vec<ContextItem>, budget: &ContextBudget) -> Self {
        // Sort: primary by priority (lower = first), secondary by relevance (higher = first)
        candidates.sort_by(|a, b| {
            a.priority
                .cmp(&b.priority)
                .then(b.relevance_score.partial_cmp(&a.relevance_score).unwrap_or(std::cmp::Ordering::Equal))
        });

        let mut items = Vec::new();
        let mut total_tokens = 0;
        let mut excluded_count = 0;

        for candidate in candidates {
            if total_tokens + candidate.estimated_tokens <= budget.max_tokens {
                total_tokens += candidate.estimated_tokens;
                items.push(candidate);
            } else {
                excluded_count += 1;
            }
        }

        Self {
            items,
            total_tokens,
            excluded_count,
            budget_max: budget.max_tokens,
        }
    }

    /// Render the assembled context as a formatted string with section headers.
    pub fn render(&self) -> String {
        if self.items.is_empty() {
            return String::new();
        }

        let mut output = String::new();
        let mut current_priority = None;

        for item in &self.items {
            if current_priority != Some(item.priority) {
                if !output.is_empty() {
                    output.push('\n');
                }
                let section = priority_label(item.priority);
                output.push_str(&format!("## {section}\n\n"));
                current_priority = Some(item.priority);
            }

            output.push_str(&item.content);
            output.push('\n');
        }

        output
    }

    /// Whether the budget was fully utilized (some items were excluded).
    pub fn is_truncated(&self) -> bool {
        self.excluded_count > 0
    }

    /// Remaining token budget.
    pub fn remaining_tokens(&self) -> usize {
        self.budget_max.saturating_sub(self.total_tokens)
    }
}

fn priority_label(priority: u8) -> &'static str {
    match priority {
        0 => "Critical Context",
        1..=10 => "Retry Context",
        11..=15 => "Specifications",
        16..=25 => "Similar Tasks",
        26..=40 => "Learnings",
        _ => "Historical Context",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(id: i64, text: &str, priority: u8, tokens: usize, score: f32) -> ContextItem {
        ContextItem {
            memory_id: id,
            content: text.to_string(),
            priority,
            estimated_tokens: tokens,
            relevance_score: score,
            memory_type: MemoryType::Semantic,
            category: None,
        }
    }

    #[test]
    fn budget_token_estimation() {
        let budget = ContextBudget::new(1000);
        assert_eq!(budget.estimate_tokens("hello"), 2); // 5 * 0.25 = 1.25, ceil = 2
        assert_eq!(budget.estimate_tokens(""), 0);
    }

    #[test]
    fn assemble_within_budget() {
        let budget = ContextBudget::new(100);
        let candidates = vec![
            item(1, "first", 0, 30, 1.0),
            item(2, "second", 10, 30, 0.8),
            item(3, "third", 25, 30, 0.5),
        ];

        let assembly = ContextAssembly::assemble(candidates, &budget);
        assert_eq!(assembly.items.len(), 3);
        assert_eq!(assembly.total_tokens, 90);
        assert_eq!(assembly.excluded_count, 0);
        assert!(!assembly.is_truncated());
    }

    #[test]
    fn assemble_exceeds_budget() {
        let budget = ContextBudget::new(50);
        let candidates = vec![
            item(1, "high priority", 0, 30, 1.0),
            item(2, "medium priority", 25, 30, 0.8),
            item(3, "low priority", 60, 30, 0.5),
        ];

        let assembly = ContextAssembly::assemble(candidates, &budget);
        assert_eq!(assembly.items.len(), 1); // only first fits
        assert_eq!(assembly.total_tokens, 30);
        assert_eq!(assembly.excluded_count, 2);
        assert!(assembly.is_truncated());
    }

    #[test]
    fn priority_ordering() {
        let budget = ContextBudget::new(1000);
        let candidates = vec![
            item(1, "low", 60, 10, 1.0),
            item(2, "high", 0, 10, 0.5),
            item(3, "mid", 25, 10, 0.8),
        ];

        let assembly = ContextAssembly::assemble(candidates, &budget);
        assert_eq!(assembly.items[0].memory_id, 2); // priority 0
        assert_eq!(assembly.items[1].memory_id, 3); // priority 25
        assert_eq!(assembly.items[2].memory_id, 1); // priority 60
    }

    #[test]
    fn relevance_within_priority() {
        let budget = ContextBudget::new(1000);
        let candidates = vec![
            item(1, "low relevance", 25, 10, 0.3),
            item(2, "high relevance", 25, 10, 0.9),
            item(3, "mid relevance", 25, 10, 0.6),
        ];

        let assembly = ContextAssembly::assemble(candidates, &budget);
        assert_eq!(assembly.items[0].memory_id, 2); // highest relevance
        assert_eq!(assembly.items[1].memory_id, 3);
        assert_eq!(assembly.items[2].memory_id, 1);
    }

    #[test]
    fn render_with_sections() {
        let budget = ContextBudget::new(1000);
        let candidates = vec![
            item(1, "critical info", 0, 10, 1.0),
            item(2, "a learning", 40, 10, 0.5),
        ];

        let assembly = ContextAssembly::assemble(candidates, &budget);
        let rendered = assembly.render();
        assert!(rendered.contains("## Critical Context"));
        assert!(rendered.contains("## Learnings"));
        assert!(rendered.contains("critical info"));
        assert!(rendered.contains("a learning"));
    }

    #[test]
    fn empty_assembly() {
        let budget = ContextBudget::new(1000);
        let assembly = ContextAssembly::assemble(Vec::new(), &budget);
        assert!(assembly.items.is_empty());
        assert_eq!(assembly.render(), "");
        assert!(!assembly.is_truncated());
        assert_eq!(assembly.remaining_tokens(), 1000);
    }

    #[test]
    fn skip_too_large_items() {
        let budget = ContextBudget::new(50);
        let candidates = vec![
            item(1, "small", 0, 20, 1.0),
            item(2, "too big", 10, 100, 0.9), // doesn't fit
            item(3, "also small", 25, 20, 0.5), // fits after skipping big
        ];

        let assembly = ContextAssembly::assemble(candidates, &budget);
        assert_eq!(assembly.items.len(), 2);
        assert_eq!(assembly.items[0].memory_id, 1);
        assert_eq!(assembly.items[1].memory_id, 3);
        assert_eq!(assembly.excluded_count, 1);
    }
}

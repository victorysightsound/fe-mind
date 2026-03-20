use chrono::{DateTime, Duration, Utc};

/// Result of query expansion — cleaned text plus any extracted date filters.
#[derive(Debug, Clone)]
pub struct ExpandedQuery {
    /// The query text with temporal expressions removed.
    pub cleaned_text: String,
    /// SQL date-range filters extracted from temporal expressions.
    pub date_filters: Vec<DateFilter>,
}

/// A date range filter extracted from a query.
#[derive(Debug, Clone)]
pub struct DateFilter {
    /// SQL column to filter on.
    pub column: String,
    /// Filter operation.
    pub operator: FilterOp,
}

/// Date filter operation.
#[derive(Debug, Clone)]
pub enum FilterOp {
    /// created_at > datetime
    After(DateTime<Utc>),
    /// created_at < datetime
    Before(DateTime<Utc>),
    /// created_at BETWEEN start AND end
    Between(DateTime<Utc>, DateTime<Utc>),
}

/// Expand temporal expressions in a query into date filters.
///
/// Converts natural language time references like "last week", "yesterday",
/// "last month" into SQL-compatible date range filters.
pub fn expand_query(query: &str, now: DateTime<Utc>) -> ExpandedQuery {
    let lower = query.to_lowercase();
    let mut cleaned = query.to_string();
    let mut filters = Vec::new();

    let patterns: &[(&str, Box<dyn Fn(DateTime<Utc>) -> FilterOp>)] = &[
        ("yesterday", Box::new(|now| {
            FilterOp::Between(now - Duration::days(2), now - Duration::days(1))
        })),
        ("last week", Box::new(|now| {
            FilterOp::After(now - Duration::days(7))
        })),
        ("last month", Box::new(|now| {
            FilterOp::After(now - Duration::days(30))
        })),
        ("last year", Box::new(|now| {
            FilterOp::After(now - Duration::days(365))
        })),
        ("today", Box::new(|now| {
            FilterOp::After(now - Duration::days(1))
        })),
        ("this week", Box::new(|now| {
            FilterOp::After(now - Duration::days(7))
        })),
        ("this month", Box::new(|now| {
            FilterOp::After(now - Duration::days(30))
        })),
        ("recently", Box::new(|now| {
            FilterOp::After(now - Duration::days(7))
        })),
    ];

    for (pattern, make_filter) in patterns {
        if lower.contains(pattern) {
            filters.push(DateFilter {
                column: "created_at".to_string(),
                operator: make_filter(now),
            });
            // Remove the temporal expression from the query
            if let Some(pos) = lower.find(pattern) {
                cleaned = format!(
                    "{} {}",
                    cleaned[..pos].trim(),
                    cleaned[pos + pattern.len()..].trim()
                ).trim().to_string();
            }
            break; // Only apply first match
        }
    }

    ExpandedQuery {
        cleaned_text: cleaned,
        date_filters: filters,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_temporal_expression() {
        let now = Utc::now();
        let result = expand_query("authentication error JWT", now);
        assert_eq!(result.cleaned_text, "authentication error JWT");
        assert!(result.date_filters.is_empty());
    }

    #[test]
    fn last_week_expansion() {
        let now = Utc::now();
        let result = expand_query("errors from last week", now);
        assert_eq!(result.cleaned_text, "errors from");
        assert_eq!(result.date_filters.len(), 1);
        assert!(matches!(result.date_filters[0].operator, FilterOp::After(_)));
    }

    #[test]
    fn yesterday_expansion() {
        let now = Utc::now();
        let result = expand_query("what happened yesterday", now);
        assert!(result.date_filters.len() == 1);
        assert!(matches!(result.date_filters[0].operator, FilterOp::Between(_, _)));
    }

    #[test]
    fn last_month_expansion() {
        let now = Utc::now();
        let result = expand_query("decisions last month", now);
        assert_eq!(result.date_filters.len(), 1);
        assert_eq!(result.date_filters[0].column, "created_at");
    }

    #[test]
    fn cleaned_text_usable_for_fts() {
        let now = Utc::now();
        let result = expand_query("build errors from last week in production", now);
        // The cleaned text should still be useful for FTS search
        assert!(result.cleaned_text.contains("build errors"));
        assert!(result.cleaned_text.contains("production"));
        assert!(!result.cleaned_text.contains("last week"));
    }
}

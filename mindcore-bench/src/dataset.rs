use serde::{Deserialize, Serialize};

/// A single conversation turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Turn {
    pub role: String,
    pub content: String,
    /// Evidence turns are marked with this flag.
    #[serde(default)]
    pub has_answer: bool,
}

/// A conversation session (list of turns).
pub type Session = Vec<Turn>;

/// The 6 base question types in LongMemEval.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QuestionType {
    #[serde(rename = "single-session-user")]
    SingleSessionUser,
    #[serde(rename = "single-session-assistant")]
    SingleSessionAssistant,
    #[serde(rename = "single-session-preference")]
    SingleSessionPreference,
    #[serde(rename = "multi-session")]
    MultiSession,
    #[serde(rename = "knowledge-update")]
    KnowledgeUpdate,
    #[serde(rename = "temporal-reasoning")]
    TemporalReasoning,
}

impl QuestionType {
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::SingleSessionUser => "Single-Session (User)",
            Self::SingleSessionAssistant => "Single-Session (Assistant)",
            Self::SingleSessionPreference => "Single-Session (Preference)",
            Self::MultiSession => "Multi-Session",
            Self::KnowledgeUpdate => "Knowledge Update",
            Self::TemporalReasoning => "Temporal Reasoning",
        }
    }
}

/// Ground truth answer — can be a string or array of strings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Answer {
    Single(String),
    Multiple(Vec<String>),
}

impl Answer {
    /// Flatten to a single string for judge comparison.
    pub fn as_text(&self) -> String {
        match self {
            Self::Single(s) => s.clone(),
            Self::Multiple(v) => v.join("; "),
        }
    }
}

/// A single LongMemEval question entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Question {
    pub question_id: String,
    pub question_type: QuestionType,
    pub question: String,
    pub answer: Answer,
    pub question_date: String,
    #[serde(default)]
    pub haystack_session_ids: Vec<String>,
    #[serde(default)]
    pub haystack_dates: Vec<String>,
    pub haystack_sessions: Vec<Session>,
    #[serde(default)]
    pub answer_session_ids: Vec<String>,
}

impl Question {
    /// Whether this is an abstention question (should be answered with "I don't know").
    pub fn is_abstention(&self) -> bool {
        self.question_id.contains("_abs")
    }

    /// Total number of turns across all haystack sessions.
    pub fn total_turns(&self) -> usize {
        self.haystack_sessions.iter().map(|s| s.len()).sum()
    }
}

/// The full LongMemEval dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Dataset {
    pub questions: Vec<Question>,
}

impl Dataset {
    /// Parse from JSON bytes.
    pub fn from_json(data: &[u8]) -> anyhow::Result<Self> {
        let questions: Vec<Question> = serde_json::from_slice(data)?;
        Ok(Self { questions })
    }

    /// Load from a file path.
    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        let data = std::fs::read(path)?;
        Self::from_json(&data)
    }

    /// Count by question type.
    pub fn count_by_type(&self) -> std::collections::HashMap<QuestionType, usize> {
        let mut counts = std::collections::HashMap::new();
        for q in &self.questions {
            *counts.entry(q.question_type).or_insert(0) += 1;
        }
        counts
    }

    /// Count abstention questions.
    pub fn abstention_count(&self) -> usize {
        self.questions.iter().filter(|q| q.is_abstention()).count()
    }
}

/// Result of evaluating a single question.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalResult {
    pub question_id: String,
    pub question_type: QuestionType,
    pub is_abstention: bool,
    pub hypothesis: String,
    pub ground_truth: String,
    pub is_correct: bool,
    pub tokens_used: u32,
}

/// Dataset variant (oracle, S, M).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatasetVariant {
    /// Oracle: evidence sessions only (~15MB)
    Oracle,
    /// S: ~40 sessions, ~115K tokens (~277MB)
    Small,
    /// M: ~500 sessions, ~1.5M tokens (~2.7GB)
    Medium,
}

impl DatasetVariant {
    pub fn filename(&self) -> &'static str {
        match self {
            Self::Oracle => "longmemeval_oracle.json",
            Self::Small => "longmemeval_s_cleaned.json",
            Self::Medium => "longmemeval_m_cleaned.json",
        }
    }

    pub fn download_url(&self) -> String {
        format!(
            "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/{}",
            self.filename()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn question_type_deserialize() {
        let json = r#""single-session-user""#;
        let qt: QuestionType = serde_json::from_str(json).unwrap();
        assert_eq!(qt, QuestionType::SingleSessionUser);
    }

    #[test]
    fn answer_single() {
        let json = r#""the answer""#;
        let a: Answer = serde_json::from_str(json).unwrap();
        assert_eq!(a.as_text(), "the answer");
    }

    #[test]
    fn answer_multiple() {
        let json = r#"["answer1", "answer2"]"#;
        let a: Answer = serde_json::from_str(json).unwrap();
        assert_eq!(a.as_text(), "answer1; answer2");
    }

    #[test]
    fn abstention_detection() {
        let q = Question {
            question_id: "q123_abs".into(),
            question_type: QuestionType::SingleSessionUser,
            question: "test".into(),
            answer: Answer::Single("n/a".into()),
            question_date: "2024/01/01 (Mon) 10:00".into(),
            haystack_session_ids: vec![],
            haystack_dates: vec![],
            haystack_sessions: vec![],
            answer_session_ids: vec![],
        };
        assert!(q.is_abstention());

        let q2 = Question {
            question_id: "q456".into(),
            ..q.clone()
        };
        assert!(!q2.is_abstention());
    }

    #[test]
    fn dataset_variant_urls() {
        let oracle = DatasetVariant::Oracle;
        assert!(oracle.download_url().contains("longmemeval_oracle.json"));
        assert_eq!(oracle.filename(), "longmemeval_oracle.json");
    }
}

use std::collections::HashMap;

use anyhow::Result;
use chrono::{DateTime, Utc};
use mindcore::engine::MemoryEngine;
use mindcore::traits::{MemoryRecord, MemoryType};
use serde::{Deserialize, Serialize};

use crate::dataset::Question;

/// A conversation turn stored as a MindCore memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationMemory {
    pub id: Option<i64>,
    pub content: String,
    pub role: String,
    pub session_index: usize,
    pub turn_index: usize,
    pub session_date: String,
    pub created_at: DateTime<Utc>,
}

impl MemoryRecord for ConversationMemory {
    fn id(&self) -> Option<i64> {
        self.id
    }

    fn searchable_text(&self) -> String {
        self.content.clone()
    }

    fn memory_type(&self) -> MemoryType {
        MemoryType::Episodic
    }

    fn importance(&self) -> u8 {
        // User messages slightly more important (they contain preferences/facts)
        if self.role == "user" { 6 } else { 5 }
    }

    fn created_at(&self) -> DateTime<Utc> {
        self.created_at
    }

    fn category(&self) -> Option<&str> {
        Some(&self.role)
    }

    fn metadata(&self) -> HashMap<String, String> {
        let mut meta = HashMap::new();
        meta.insert("session_index".into(), self.session_index.to_string());
        meta.insert("turn_index".into(), self.turn_index.to_string());
        meta.insert("session_date".into(), self.session_date.clone());
        meta
    }
}

/// Ingest all haystack sessions from a question into a MindCore engine.
///
/// Returns the number of memories stored.
pub fn ingest_question(
    engine: &MemoryEngine<ConversationMemory>,
    question: &Question,
) -> Result<usize> {
    let mut stored = 0;

    for (session_idx, session) in question.haystack_sessions.iter().enumerate() {
        let session_date = question
            .haystack_dates
            .get(session_idx)
            .cloned()
            .unwrap_or_default();

        for (turn_idx, turn) in session.iter().enumerate() {
            // Skip empty turns
            if turn.content.trim().is_empty() {
                continue;
            }

            let memory = ConversationMemory {
                id: None,
                content: turn.content.clone(),
                role: turn.role.clone(),
                session_index: session_idx,
                turn_index: turn_idx,
                session_date: session_date.clone(),
                created_at: Utc::now(),
            };

            match engine.store(&memory) {
                Ok(mindcore::memory::store::StoreResult::Added(_)) => stored += 1,
                Ok(mindcore::memory::store::StoreResult::Duplicate(_)) => {} // skip dupes
                Err(e) => {
                    tracing::warn!("Failed to store turn: {e}");
                }
            }
        }
    }

    Ok(stored)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{Answer, QuestionType, Session, Turn};

    fn test_question() -> Question {
        Question {
            question_id: "test_q1".into(),
            question_type: QuestionType::SingleSessionUser,
            question: "What is the user's name?".into(),
            answer: Answer::Single("John".into()),
            question_date: "2024/01/15 (Mon) 14:30".into(),
            haystack_session_ids: vec!["s1".into()],
            haystack_dates: vec!["2024/01/15 (Mon) 10:00".into()],
            haystack_sessions: vec![vec![
                Turn { role: "user".into(), content: "My name is John".into(), has_answer: true },
                Turn { role: "assistant".into(), content: "Nice to meet you, John!".into(), has_answer: false },
                Turn { role: "user".into(), content: "What's the weather like?".into(), has_answer: false },
                Turn { role: "assistant".into(), content: "It's sunny today.".into(), has_answer: false },
            ]],
            answer_session_ids: vec!["s1".into()],
        }
    }

    #[test]
    fn ingest_stores_all_turns() {
        let engine = MemoryEngine::<ConversationMemory>::builder()
            .build()
            .expect("build");
        let q = test_question();
        let stored = ingest_question(&engine, &q).expect("ingest");
        assert_eq!(stored, 4, "all 4 turns should be stored");
        assert_eq!(engine.count().expect("count"), 4);
    }

    #[test]
    fn ingest_searchable() {
        let engine = MemoryEngine::<ConversationMemory>::builder()
            .build()
            .expect("build");
        let q = test_question();
        ingest_question(&engine, &q).expect("ingest");

        let results = engine.search("John").execute().expect("search");
        assert!(!results.is_empty(), "should find 'John' in memories");
    }
}

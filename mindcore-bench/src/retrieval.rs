use anyhow::Result;
use mindcore::context::ContextBudget;
use mindcore::engine::MemoryEngine;

use crate::dataset::Question;
use crate::ingest::{ingest_question, ConversationMemory};

/// Context retrieved for answering a question.
pub struct RetrievedContext {
    /// The assembled context text for LLM prompt.
    pub context_text: String,
    /// Number of memories stored for this question.
    pub memories_stored: usize,
    /// Number of context items included.
    pub items_included: usize,
    /// Total tokens used.
    pub tokens_used: usize,
}

/// Process a single question: format all haystack sessions as context.
///
/// For the oracle dataset, we include all evidence sessions directly
/// since they're the minimal set needed to answer the question.
/// For S/M datasets with many distractor sessions, we'd use MindCore
/// search to select the most relevant sessions.
pub fn process_question(
    question: &Question,
    context_budget: usize,
) -> Result<RetrievedContext> {
    // For oracle dataset: format all sessions as chronological context
    // This is the "full history" baseline approach.
    // MindCore search-based retrieval can be layered on top for S/M datasets.
    let mut context_parts = Vec::new();
    let mut total_chars = 0;
    let budget_chars = (context_budget as f32 / 0.25) as usize; // tokens → chars

    for (session_idx, session) in question.haystack_sessions.iter().enumerate() {
        let date = question
            .haystack_dates
            .get(session_idx)
            .map(|d| d.as_str())
            .unwrap_or("unknown date");

        let mut session_text = format!("[Session from {date}]\n");

        for turn in session {
            let line = format!("{}: {}\n", turn.role, turn.content);
            session_text.push_str(&line);
        }

        total_chars += session_text.len();
        if total_chars > budget_chars {
            break; // Budget exceeded
        }

        context_parts.push(session_text);
    }

    let context_text = context_parts.join("\n");
    let tokens_used = (context_text.len() as f32 * 0.25) as usize;

    Ok(RetrievedContext {
        context_text,
        memories_stored: question.total_turns(),
        items_included: context_parts.len(),
        tokens_used,
    })
}

/// Build the generation prompt using the LongMemEval template.
pub fn build_generation_prompt(
    context: &str,
    question: &str,
    question_date: &str,
) -> String {
    format!(
        "I will give you several history chats between a user and an AI assistant. \
         Based on the chat history, answer the question at the end. \
         Answer the question step by step: first extract all the relevant information, \
         and then reason over the information to get the answer. \
         If the information needed to answer the question is not available in the chat history, \
         say \"I don't know\" or \"The information is not available in the chat history.\"\n\n\
         History Chats:\n\n\
         {context}\n\n\
         Current Date: {question_date}\n\
         Question: {question}\n\
         Answer (step by step):"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{Answer, QuestionType, Turn};

    fn test_question() -> Question {
        Question {
            question_id: "test_q1".into(),
            question_type: QuestionType::SingleSessionUser,
            question: "What is the user's favorite color?".into(),
            answer: Answer::Single("blue".into()),
            question_date: "2024/01/15 (Mon) 14:30".into(),
            haystack_session_ids: vec!["s1".into()],
            haystack_dates: vec!["2024/01/15 (Mon) 10:00".into()],
            haystack_sessions: vec![vec![
                Turn { role: "user".into(), content: "My favorite color is blue".into(), has_answer: true },
                Turn { role: "assistant".into(), content: "That's a nice color!".into(), has_answer: false },
            ]],
            answer_session_ids: vec!["s1".into()],
        }
    }

    #[test]
    fn process_question_retrieves_context() {
        let q = test_question();
        let result = process_question(&q, 4096).expect("process");
        assert!(result.memories_stored > 0);
        assert!(result.items_included > 0);
        assert!(!result.context_text.is_empty());
        assert!(result.context_text.contains("blue"));
    }

    #[test]
    fn generation_prompt_format() {
        let prompt = build_generation_prompt("some history", "What color?", "2024/01/15");
        assert!(prompt.contains("History Chats:"));
        assert!(prompt.contains("some history"));
        assert!(prompt.contains("What color?"));
        assert!(prompt.contains("2024/01/15"));
        assert!(prompt.contains("step by step"));
    }
}

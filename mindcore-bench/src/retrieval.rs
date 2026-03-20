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

/// Process a single question: create engine, ingest sessions, retrieve context.
///
/// Uses a fresh in-memory engine per question to avoid cross-contamination.
pub fn process_question(
    question: &Question,
    context_budget: usize,
) -> Result<RetrievedContext> {
    // Fresh engine per question
    let engine = MemoryEngine::<ConversationMemory>::builder()
        .build()?;

    // Ingest all sessions
    let memories_stored = ingest_question(&engine, question)?;

    if memories_stored == 0 {
        return Ok(RetrievedContext {
            context_text: String::new(),
            memories_stored: 0,
            items_included: 0,
            tokens_used: 0,
        });
    }

    // Retrieve context for the question
    let budget = ContextBudget::new(context_budget);
    let assembly = engine.assemble_context(&question.question, &budget)?;

    let context_text = assembly.render();

    Ok(RetrievedContext {
        context_text,
        memories_stored,
        items_included: assembly.items.len(),
        tokens_used: assembly.total_tokens,
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

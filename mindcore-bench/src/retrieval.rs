use anyhow::Result;

use crate::dataset::{Question, QuestionType};

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
/// Pass `context_budget = 0` for unlimited (recommended for oracle).
pub fn process_question(
    question: &Question,
    context_budget: usize,
) -> Result<RetrievedContext> {
    let mut context_parts = Vec::new();
    let mut total_chars = 0;
    let budget_chars = if context_budget == 0 {
        usize::MAX // unlimited
    } else {
        (context_budget as f32 / 0.25) as usize
    };

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
            break;
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

/// Build the generation prompt with type-specific instructions.
pub fn build_generation_prompt(
    context: &str,
    question: &str,
    question_date: &str,
    question_type: QuestionType,
    is_abstention: bool,
) -> String {
    let preamble = format!(
        "I will give you several history chats between a user and an AI assistant. \
         Based on the chat history, answer the question at the end.\n\n\
         History Chats:\n\n\
         {context}\n\n\
         Current Date: {question_date}\n\
         Question: {question}\n\n"
    );

    let type_instruction = if is_abstention {
        "Instructions: If the chat history does not contain information that DIRECTLY answers \
         this question, you MUST respond with \"I don't know\" or \"The information is not \
         available in the chat history.\" Do NOT attempt to infer, extrapolate, or guess. \
         Only answer if the information is explicitly stated in the conversations. \
         If you can answer, provide the answer concisely."
            .to_string()
    } else {
        match question_type {
            QuestionType::SingleSessionPreference => {
                "Instructions: Based on the chat history, describe what the user's CONTENT \
                 preferences would be when responding to this question. Focus on the TOPICS, \
                 SUBJECTS, and SPECIFIC INTERESTS the user has expressed — NOT on response \
                 formatting or structure.\n\n\
                 Your answer MUST be in the format: \"The user would prefer responses that...\"\n\n\
                 Example 1:\n\
                 Question: Can you recommend some accessories for my camera?\n\
                 Good answer: The user would prefer suggestions of Sony-compatible accessories \
                 that enhance their landscape photography, based on their discussion of the \
                 Sony Alpha camera and recent mountain photography trips. They might not prefer \
                 suggestions for other camera brands or studio photography gear.\n\n\
                 Example 2:\n\
                 Question: Can you suggest some new recipes to try?\n\
                 Good answer: The user would prefer recipes that incorporate quinoa and roasted \
                 vegetables, building on their recent success with Mediterranean-style meal prep. \
                 They might not prefer recipes with dairy, given their mention of lactose \
                 intolerance.\n\n\
                 BAD answers describe formatting preferences like \"well-organized with bullet \
                 points\" or \"detailed and comprehensive.\" Focus on WHAT the user wants to \
                 hear about, not HOW it should be formatted.\n\n\
                 Now describe the user's content preferences for the question above:"
                    .to_string()
            }
            QuestionType::TemporalReasoning => {
                format!(
                    "Instructions: Answer this question step by step. Pay close attention to \
                     dates and timestamps on each session. \
                     IMPORTANT: Before computing any count or duration, list EVERY relevant \
                     event with its exact date. Then count them explicitly (1, 2, 3...) or \
                     compute the date arithmetic step by step. Do not estimate or shortcut. \
                     When counting days between dates, enumerate each step. \
                     Current Date: {question_date}"
                )
            }
            QuestionType::KnowledgeUpdate => {
                "Instructions: Answer this question step by step. When information has been \
                 updated across sessions, use the MOST RECENT value as the primary answer. \
                 IMPORTANT: List ALL versions of the relevant information chronologically with \
                 their session dates. Then clearly state the latest/most recent value as your \
                 final answer."
                    .to_string()
            }
            QuestionType::MultiSession => {
                "Instructions: This question requires synthesizing information across multiple \
                 sessions. Answer step by step. \
                 IMPORTANT: Before giving your final answer, enumerate ALL relevant items/facts \
                 from EVERY session. Number each one explicitly. Do not skip any session. \
                 Then compile your final answer from the complete list."
                    .to_string()
            }
            _ => {
                // SingleSessionUser, SingleSessionAssistant
                "Instructions: Answer the question based on the chat history. \
                 First extract the relevant information, then provide a concise answer."
                    .to_string()
            }
        }
    };

    format!("{preamble}{type_instruction}\n\nAnswer:")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{Answer, Turn};

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
                Turn {
                    role: "user".into(),
                    content: "My favorite color is blue".into(),
                    has_answer: true,
                },
                Turn {
                    role: "assistant".into(),
                    content: "That's a nice color!".into(),
                    has_answer: false,
                },
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
    fn process_question_unlimited_budget() {
        let q = test_question();
        let result = process_question(&q, 0).expect("process");
        assert!(result.items_included > 0);
        assert!(result.context_text.contains("blue"));
    }

    #[test]
    fn generation_prompt_default() {
        let prompt = build_generation_prompt(
            "some history",
            "What color?",
            "2024/01/15",
            QuestionType::SingleSessionUser,
            false,
        );
        assert!(prompt.contains("History Chats:"));
        assert!(prompt.contains("some history"));
        assert!(prompt.contains("What color?"));
        assert!(prompt.contains("extract the relevant information"));
    }

    #[test]
    fn preference_prompt_format() {
        let prompt = build_generation_prompt(
            "some history",
            "Recommend resources?",
            "2024/01/15",
            QuestionType::SingleSessionPreference,
            false,
        );
        assert!(prompt.contains("The user would prefer responses that"));
        assert!(prompt.contains("CONTENT"));
        assert!(prompt.contains("BAD answers describe formatting"));
        // Verify few-shot examples are present
        assert!(prompt.contains("Sony-compatible"));
        assert!(prompt.contains("quinoa"));
    }

    #[test]
    fn temporal_prompt_format() {
        let prompt = build_generation_prompt(
            "some history",
            "How many days?",
            "2024/01/15",
            QuestionType::TemporalReasoning,
            false,
        );
        assert!(prompt.contains("EVERY relevant event"));
        assert!(prompt.contains("count them explicitly"));
    }

    #[test]
    fn abstention_prompt_format() {
        let prompt = build_generation_prompt(
            "some history",
            "What did I say?",
            "2024/01/15",
            QuestionType::SingleSessionUser,
            true,
        );
        assert!(prompt.contains("MUST respond with"));
        assert!(prompt.contains("Do NOT attempt to infer"));
    }
}

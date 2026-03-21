use anyhow::Result;

use crate::dataset::QuestionType;
use crate::llm::ClaudeCliClient;

/// Question types that benefit from self-verification.
///
/// These involve counting, arithmetic, or multi-item enumeration
/// where the model's chain-of-thought may produce the right steps
/// but arrive at the wrong final number.
fn needs_verification(question_type: QuestionType, is_abstention: bool) -> bool {
    if is_abstention {
        return false; // abstention answers are yes/no, nothing to verify
    }
    matches!(
        question_type,
        QuestionType::MultiSession | QuestionType::KnowledgeUpdate
    )
}

/// Optionally verify a hypothesis by asking the model to re-check its work.
///
/// Returns the original hypothesis unchanged for question types that don't
/// need verification. For multi-session and temporal reasoning questions,
/// makes a second LLM call to verify counts and computations.
///
/// The `context` parameter provides the original chat history so the verifier
/// can check claims against the source material rather than hallucinating.
pub fn maybe_verify(
    client: &ClaudeCliClient,
    context: &str,
    question: &str,
    hypothesis: &str,
    question_type: QuestionType,
    is_abstention: bool,
) -> Result<String> {
    if !needs_verification(question_type, is_abstention) {
        return Ok(hypothesis.to_string());
    }

    let verify_prompt = build_verify_prompt(context, question, hypothesis, question_type);
    let verified = client.complete(&verify_prompt, 512)?;

    // If verification produced a non-empty response, use it.
    // Otherwise fall back to original.
    if verified.trim().is_empty() {
        Ok(hypothesis.to_string())
    } else {
        Ok(verified)
    }
}

fn build_verify_prompt(
    context: &str,
    question: &str,
    hypothesis: &str,
    question_type: QuestionType,
) -> String {
    let type_check = match question_type {
        QuestionType::MultiSession => {
            "Your job: if the answer involves a count, re-enumerate every relevant item \
             from the chat history numbered 1, 2, 3... and recount. If the answer is a \
             list, verify every item from the chat history is accounted for and none are \
             duplicated or missing. If it involves a sum, recompute the arithmetic."
        }
        QuestionType::TemporalReasoning => {
            "Your job: if the answer involves date arithmetic, recompute the \
             number of days/weeks/months between the dates step by step using the \
             chat history timestamps. If it involves ordering events, re-list each \
             event with its date from the chat history and re-sort."
        }
        QuestionType::KnowledgeUpdate => {
            "Your job: verify the answer uses the value from the MOST RECENT \
             session in the chat history. Re-list all versions chronologically \
             from the chat history and confirm the final one is correct."
        }
        _ => "",
    };

    format!(
        "You are verifying an answer. You have the original chat history and the \
         model's answer. Check whether the final answer is correct by going back \
         to the source material.\n\n\
         Chat History:\n{context}\n\n\
         Question: {question}\n\n\
         Model's answer:\n{hypothesis}\n\n\
         {type_check}\n\n\
         If the answer is correct, output it again unchanged. \
         If you find an error (wrong count, wrong arithmetic, wrong item, wrong \
         version), output ONLY the corrected final answer — no explanation, no \
         reasoning, just the answer."
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_session_skips_verification() {
        assert!(!needs_verification(
            QuestionType::SingleSessionUser,
            false
        ));
        assert!(!needs_verification(
            QuestionType::SingleSessionAssistant,
            false
        ));
        assert!(!needs_verification(
            QuestionType::SingleSessionPreference,
            false
        ));
    }

    #[test]
    fn multi_session_needs_verification() {
        assert!(needs_verification(QuestionType::MultiSession, false));
        assert!(needs_verification(QuestionType::KnowledgeUpdate, false));
        // Temporal reasoning excluded — verification causes regressions
        assert!(!needs_verification(QuestionType::TemporalReasoning, false));
    }

    #[test]
    fn abstention_skips_verification() {
        assert!(!needs_verification(QuestionType::MultiSession, true));
        assert!(!needs_verification(QuestionType::TemporalReasoning, true));
    }

    #[test]
    fn verify_prompt_contains_type_check() {
        let ctx = "some chat history";
        let prompt =
            build_verify_prompt(ctx, "How many?", "I counted 5 items.", QuestionType::MultiSession);
        assert!(prompt.contains("re-enumerate"));
        assert!(prompt.contains("recount"));
        assert!(prompt.contains("Chat History:"));

        let prompt =
            build_verify_prompt(ctx, "How long?", "14 days.", QuestionType::TemporalReasoning);
        assert!(prompt.contains("recompute"));

        let prompt =
            build_verify_prompt(ctx, "What now?", "The new value.", QuestionType::KnowledgeUpdate);
        assert!(prompt.contains("MOST RECENT"));
    }
}

use anyhow::Result;

use crate::dataset::QuestionType;
use crate::llm::ClaudeClient;

/// Judge a hypothesis against the ground truth using type-specific prompts.
///
/// Returns (is_correct, tokens_used).
pub async fn judge_answer(
    client: &ClaudeClient,
    question: &str,
    ground_truth: &str,
    hypothesis: &str,
    question_type: QuestionType,
    is_abstention: bool,
) -> Result<(bool, u32)> {
    let judge_instruction = if is_abstention {
        "You are evaluating a conversational AI assistant's response. \
         The question is designed to be unanswerable based on the chat history. \
         Answer \"yes\" if the model correctly identifies the question as unanswerable \
         or says it doesn't have enough information. Answer \"no\" if the model attempts \
         to provide a specific answer."
    } else {
        match question_type {
            QuestionType::TemporalReasoning => {
                "You are evaluating a conversational AI assistant's response. \
                 Answer \"yes\" if the response contains the correct answer or is equivalent \
                 to the correct answer or contains all intermediate steps needed to reach the answer. \
                 Do not penalize off-by-one errors for the number of days, weeks, or months. \
                 Answer \"no\" otherwise."
            }
            QuestionType::KnowledgeUpdate => {
                "You are evaluating a conversational AI assistant's response. \
                 The response should be considered correct as long as the updated answer \
                 (the required answer) is present. It is acceptable if the response also \
                 mentions the old information, as long as the new/updated information is \
                 the primary answer. Answer \"yes\" if correct, \"no\" otherwise."
            }
            QuestionType::SingleSessionPreference => {
                "You are evaluating a conversational AI assistant's response. \
                 Answer \"yes\" if the response correctly recalls and utilizes the user's \
                 personal information and satisfies the desired response. \
                 Answer \"no\" otherwise."
            }
            _ => {
                "You are evaluating a conversational AI assistant's response. \
                 Answer \"yes\" if the response contains the correct answer or is equivalent \
                 to the correct answer or contains all intermediate steps needed to reach the answer. \
                 Answer \"no\" otherwise."
            }
        }
    };

    let prompt = format!(
        "{judge_instruction}\n\n\
         Question: {question}\n\
         Required Answer: {ground_truth}\n\
         Model's Response: {hypothesis}\n\n\
         Is the model's response correct? Answer only \"yes\" or \"no\"."
    );

    let (response, tokens) = client.complete(&prompt, 10).await?;
    let is_correct = response.to_lowercase().contains("yes");

    Ok((is_correct, tokens))
}

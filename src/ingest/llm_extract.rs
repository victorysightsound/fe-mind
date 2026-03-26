//! LLM-powered fact extraction from raw text.
//!
//! Takes raw conversation text + an LlmCallback, crafts an extraction prompt,
//! calls the LLM, and parses the response into structured ExtractedFact items.
//!
//! Feature-gated behind `llm-ingest`.

use crate::error::Result;
use crate::traits::LlmCallback;

/// A fact extracted by the LLM from raw text.
#[derive(Debug, Clone)]
pub struct ExtractedFact {
    /// The fact statement (e.g., "User's favorite food is sushi")
    pub text: String,
    /// Category: fact, decision, preference, note, lesson
    pub category: String,
    /// Importance score 1-10
    pub importance: u8,
    /// Named entities mentioned in this fact
    pub entities: Vec<String>,
    /// Relationship triples: (subject, relation, object)
    pub relationships: Vec<(String, String, String)>,
}

/// Result of an extraction operation.
#[derive(Debug, Clone)]
pub struct ExtractionResult {
    /// Extracted facts
    pub facts: Vec<ExtractedFact>,
    /// Number of LLM tokens used (approximate)
    pub tokens_used: usize,
}

/// Extract facts from raw text using an LLM.
///
/// Crafts a structured extraction prompt, calls the LLM, parses the response.
/// Handles malformed LLM output gracefully — extracts what it can.
pub fn extract_facts(text: &str, llm: &dyn LlmCallback) -> Result<ExtractionResult> {
    if text.trim().is_empty() {
        return Ok(ExtractionResult {
            facts: Vec::new(),
            tokens_used: 0,
        });
    }

    let prompt = build_extraction_prompt(text);
    let response = llm.generate(&prompt, 4096)?;
    let facts = parse_extraction_response(&response);
    let tokens_used = (prompt.len() + response.len()) / 4; // rough estimate

    Ok(ExtractionResult { facts, tokens_used })
}

/// Build the extraction prompt for the LLM.
fn build_extraction_prompt(text: &str) -> String {
    format!(
        r#"Extract structured facts from the following text. For each fact, output one line in this exact format:

CATEGORY|IMPORTANCE|FACT_TEXT|ENTITIES|RELATIONSHIPS

Where:
- CATEGORY: one of: fact, decision, preference, note, lesson
- IMPORTANCE: 1-10 (10 = critical, 1 = trivial)
- FACT_TEXT: the extracted fact as a clear statement
- ENTITIES: comma-separated entity names (people, places, things)
- RELATIONSHIPS: semicolon-separated triples as "subject>relation>object"

Rules:
- Extract EVERY concrete fact, decision, preference, and noteworthy statement
- Each fact should be independently understandable
- Include entity names exactly as mentioned in the text
- For relationships, use clear relation types: married_to, works_at, citizen_of, author_of, likes, dislikes, lives_in, born_in, created_by, member_of, supersedes, etc.
- If a fact contradicts or updates a previous fact, note the relationship as "new_value>supersedes>old_value"
- If no entities or relationships, leave those fields empty
- Output ONLY the formatted lines, no explanations or headers

TEXT:
{text}

EXTRACTED FACTS:"#
    )
}

/// Parse the LLM's extraction response into structured facts.
///
/// Handles malformed lines gracefully — extracts what it can, skips the rest.
fn parse_extraction_response(response: &str) -> Vec<ExtractedFact> {
    let mut facts = Vec::new();

    for line in response.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("EXTRACTED") {
            continue;
        }

        let parts: Vec<&str> = line.splitn(5, '|').collect();
        if parts.len() < 3 {
            // Try to salvage as a simple fact with no metadata
            if !line.is_empty() && line.len() > 10 {
                facts.push(ExtractedFact {
                    text: line.to_string(),
                    category: "fact".to_string(),
                    importance: 5,
                    entities: Vec::new(),
                    relationships: Vec::new(),
                });
            }
            continue;
        }

        let category = parts[0].trim().to_lowercase();
        let importance: u8 = parts[1].trim().parse().unwrap_or(5).clamp(1, 10);
        let text = parts[2].trim().to_string();

        let entities = if parts.len() > 3 && !parts[3].trim().is_empty() {
            parts[3]
                .split(',')
                .map(|e| e.trim().to_string())
                .filter(|e| !e.is_empty())
                .collect()
        } else {
            Vec::new()
        };

        let relationships = if parts.len() > 4 && !parts[4].trim().is_empty() {
            parts[4]
                .split(';')
                .filter_map(|r| {
                    let triple: Vec<&str> = r.split('>').collect();
                    if triple.len() == 3 {
                        Some((
                            triple[0].trim().to_string(),
                            triple[1].trim().to_string(),
                            triple[2].trim().to_string(),
                        ))
                    } else {
                        None
                    }
                })
                .collect()
        } else {
            Vec::new()
        };

        if !text.is_empty() {
            facts.push(ExtractedFact {
                text,
                category: if ["fact", "decision", "preference", "note", "lesson"]
                    .contains(&category.as_str())
                {
                    category
                } else {
                    "fact".to_string()
                },
                importance,
                entities,
                relationships,
            });
        }
    }

    facts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_prompt_includes_text() {
        let prompt = build_extraction_prompt("Hello world");
        assert!(prompt.contains("Hello world"));
        assert!(prompt.contains("CATEGORY|IMPORTANCE|FACT_TEXT"));
    }

    #[test]
    fn parse_well_formed_response() {
        let response = "fact|7|User likes sushi|User,sushi|User>likes>sushi\n\
                         decision|8|Chose Rust over Python|Rust,Python|Rust>chosen_over>Python\n\
                         preference|6|Prefers dark mode||";

        let facts = parse_extraction_response(response);
        assert_eq!(facts.len(), 3);

        assert_eq!(facts[0].category, "fact");
        assert_eq!(facts[0].importance, 7);
        assert_eq!(facts[0].text, "User likes sushi");
        assert_eq!(facts[0].entities, vec!["User", "sushi"]);
        assert_eq!(facts[0].relationships.len(), 1);
        assert_eq!(
            facts[0].relationships[0],
            ("User".into(), "likes".into(), "sushi".into())
        );

        assert_eq!(facts[1].category, "decision");
        assert_eq!(facts[1].importance, 8);
        assert_eq!(facts[2].category, "preference");
    }

    #[test]
    fn parse_malformed_response() {
        let response = "fact|5|Valid fact||\n\
                         this is not formatted correctly but is long enough\n\
                         \n\
                         fact|3|Another fact||";

        let facts = parse_extraction_response(response);
        assert_eq!(facts.len(), 3); // 2 well-formed + 1 salvaged
    }

    #[test]
    fn parse_empty_response() {
        let facts = parse_extraction_response("");
        assert!(facts.is_empty());
    }

    #[test]
    fn parse_clamps_importance() {
        let response = "fact|15|Too high||\nfact|0|Too low||";
        let facts = parse_extraction_response(response);
        assert_eq!(facts[0].importance, 10); // clamped to max
        assert_eq!(facts[1].importance, 1); // clamped to min
    }

    #[test]
    fn extract_with_mock_llm() {
        struct MockLlm;
        impl LlmCallback for MockLlm {
            fn generate(&self, _prompt: &str, _max_tokens: usize) -> Result<String> {
                Ok(
                    "fact|7|User's favorite color is blue|User|User>favorite_color>blue"
                        .to_string(),
                )
            }
        }

        let result = extract_facts("I really like blue!", &MockLlm).unwrap();
        assert_eq!(result.facts.len(), 1);
        assert_eq!(result.facts[0].text, "User's favorite color is blue");
        assert_eq!(result.facts[0].relationships[0].2, "blue");
    }

    #[test]
    fn extract_empty_text() {
        struct MockLlm;
        impl LlmCallback for MockLlm {
            fn generate(&self, _: &str, _: usize) -> Result<String> {
                panic!("should not be called for empty text");
            }
        }

        let result = extract_facts("", &MockLlm).unwrap();
        assert!(result.facts.is_empty());
    }
}

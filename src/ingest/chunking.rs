/// Session-aware chunking for conversation turns.
///
/// Concatenates consecutive turns into larger chunks (~500-1000 chars) to:
/// 1. Reduce embedding count (fewer, higher-quality vectors)
/// 2. Give each embedding more semantic context
/// 3. Filter noise turns ("Thanks!", "Ok", etc.)

/// A chunk of concatenated conversation turns.
#[derive(Debug, Clone)]
pub struct SessionChunk {
    /// Concatenated text with role prefixes and session date.
    pub text: String,
    /// Number of turns included in this chunk.
    pub turn_count: usize,
}

/// Chunk a session's turns into segments of approximately `target_chars` characters.
///
/// - Filters out turns shorter than `min_turn_chars` (noise removal)
/// - Prepends session date to each chunk for temporal grounding
/// - Includes role prefixes ("User:", "Assistant:") in the text
///
/// # Arguments
/// * `turns` - Iterator of (role, content) pairs
/// * `session_date` - Date string to prepend (e.g., "2024/01/15")
/// * `target_chars` - Target chunk size in characters (~500-1000 recommended)
/// * `min_turn_chars` - Minimum turn length to include (10 recommended)
pub fn chunk_session<'a>(
    turns: impl Iterator<Item = (&'a str, &'a str)>,
    session_date: &str,
    target_chars: usize,
    min_turn_chars: usize,
) -> Vec<SessionChunk> {
    let mut chunks = Vec::new();
    let mut current_text = String::new();
    let mut current_turns = 0usize;
    let date_prefix = if session_date.is_empty() {
        String::new()
    } else {
        format!("[Session from {session_date}]\n")
    };

    for (role, content) in turns {
        // Filter noise turns
        if content.trim().len() < min_turn_chars {
            continue;
        }

        let line = format!("{role}: {content}\n");

        // If adding this turn would exceed target, flush current chunk
        if !current_text.is_empty() && current_text.len() + line.len() > target_chars {
            chunks.push(SessionChunk {
                text: format!("{date_prefix}{current_text}"),
                turn_count: current_turns,
            });
            // Overlap: carry last ~100 chars into next chunk to prevent boundary info loss
            let overlap_size = 100.min(current_text.len());
            let overlap_start = current_text.len() - overlap_size;
            // Find a word boundary for clean overlap
            let overlap_pos = current_text[overlap_start..].find(' ')
                .map(|p| overlap_start + p + 1)
                .unwrap_or(overlap_start);
            let overlap = current_text[overlap_pos..].to_string();
            current_text.clear();
            current_text.push_str(&overlap);
            current_turns = 0; // overlap turns aren't counted as new
        }

        current_text.push_str(&line);
        current_turns += 1;
    }

    // Flush remaining
    if !current_text.is_empty() {
        chunks.push(SessionChunk {
            text: format!("{date_prefix}{current_text}"),
            turn_count: current_turns,
        });
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_chunking() {
        let turns = vec![
            ("User", "What is the weather like today in San Francisco?"),
            ("Assistant", "The weather in San Francisco today is partly cloudy with temperatures around 62°F."),
            ("User", "Thanks"),  // short, will be filtered
            ("User", "What about tomorrow?"),
            ("Assistant", "Tomorrow is expected to be sunny with highs near 68°F and light winds."),
        ];

        let chunks = chunk_session(
            turns.iter().map(|(r, c)| (*r, *c)),
            "2024/01/15",
            500,
            10,
        );

        // "Thanks" should be filtered (< 10 chars)
        let total_turns: usize = chunks.iter().map(|c| c.turn_count).sum();
        assert_eq!(total_turns, 4, "should exclude 'Thanks' turn");

        // All chunks should have session date prefix
        for chunk in &chunks {
            assert!(chunk.text.contains("[Session from 2024/01/15]"));
        }
    }

    #[test]
    fn respects_target_size() {
        // Create many turns that exceed target
        let turns: Vec<(&str, &str)> = (0..20)
            .map(|_| ("User", "This is a moderately long turn that contains enough text to be meaningful for embedding purposes and search quality."))
            .collect();

        let chunks = chunk_session(
            turns.iter().map(|(r, c)| (*r, *c)),
            "2024/03/01",
            300,
            10,
        );

        assert!(chunks.len() > 1, "should split into multiple chunks");
        // Each chunk (minus date prefix) should be around target size
        for chunk in &chunks {
            // Allow some overflow since we don't split mid-turn
            assert!(chunk.text.len() < 600, "chunk too large: {} chars", chunk.text.len());
        }
    }

    #[test]
    fn filters_short_turns() {
        let turns = vec![
            ("User", "Ok"),
            ("Assistant", "Sure"),
            ("User", "Hmm"),
            ("User", "What is the capital of France and why is it important?"),
        ];

        let chunks = chunk_session(
            turns.iter().map(|(r, c)| (*r, *c)),
            "",
            500,
            10,
        );

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].turn_count, 1, "only one turn should survive filtering");
        assert!(chunks[0].text.contains("capital of France"));
    }

    #[test]
    fn empty_session() {
        let turns: Vec<(&str, &str)> = vec![];
        let chunks = chunk_session(turns.into_iter(), "2024/01/01", 500, 10);
        assert!(chunks.is_empty());
    }

    #[test]
    fn empty_date() {
        let turns = vec![("User", "This is a test message with enough content")];
        let chunks = chunk_session(
            turns.iter().map(|(r, c)| (*r, *c)),
            "",
            500,
            10,
        );
        assert_eq!(chunks.len(), 1);
        assert!(!chunks[0].text.contains("[Session from"));
    }
}

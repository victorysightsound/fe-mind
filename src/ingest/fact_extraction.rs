//! Fact extraction from structured fact lists (e.g., MemoryAgentBench).
//!
//! Parses numbered fact statements into entity-relationship triples:
//! "322. Amala Paul is a citizen of Belgium." → (Amala Paul, citizen_of, Belgium)
//!
//! Detects conflicting facts (same subject+relation, different object) and
//! creates SupersededBy graph edges between the old and new facts.

use std::collections::HashMap;

/// An extracted fact triple.
#[derive(Debug, Clone)]
pub struct FactTriple {
    /// Position in the original fact list (higher = more recent).
    pub index: usize,
    /// Subject entity.
    pub subject: String,
    /// Relationship type.
    pub relation: String,
    /// Object entity.
    pub object: String,
    /// Original fact text.
    pub original: String,
}

/// A detected conflict where a later fact supersedes an earlier one.
#[derive(Debug, Clone)]
pub struct FactConflict {
    /// The earlier (outdated) fact.
    pub old_fact: FactTriple,
    /// The later (current) fact.
    pub new_fact: FactTriple,
}

/// Parse a fact list into triples and detect conflicts.
///
/// Input: text containing numbered facts like "0. Subject relation Object."
/// Returns: (triples, conflicts)
pub fn extract_facts(text: &str) -> (Vec<FactTriple>, Vec<FactConflict>) {
    let lines: Vec<&str> = text
        .lines()
        .map(|l| l.trim())
        .filter(|l| {
            !l.is_empty()
                && l.chars()
                    .next()
                    .map(|c| c.is_ascii_digit())
                    .unwrap_or(false)
        })
        .collect();

    let mut triples = Vec::new();

    for line in &lines {
        if let Some(triple) = parse_fact_line(line) {
            triples.push(triple);
        }
    }

    // Detect conflicts: same (subject, relation) with different objects
    let mut by_key: HashMap<(String, String), Vec<&FactTriple>> = HashMap::new();
    for triple in &triples {
        let key = (triple.subject.to_lowercase(), triple.relation.clone());
        by_key.entry(key).or_default().push(triple);
    }

    let mut conflicts = Vec::new();
    for facts in by_key.values() {
        if facts.len() > 1 {
            // Sort by index (earliest first)
            let mut sorted: Vec<&&FactTriple> = facts.iter().collect();
            sorted.sort_by_key(|f| f.index);

            // Each consecutive pair is a conflict (later supersedes earlier)
            for pair in sorted.windows(2) {
                if pair[0].object.to_lowercase() != pair[1].object.to_lowercase() {
                    conflicts.push(FactConflict {
                        old_fact: (*pair[0]).clone(),
                        new_fact: (*pair[1]).clone(),
                    });
                }
            }
        }
    }

    (triples, conflicts)
}

/// Parse a single fact line into a triple.
fn parse_fact_line(line: &str) -> Option<FactTriple> {
    // Extract the index number
    let dot_pos = line.find('.')?;
    let index: usize = line[..dot_pos].trim().parse().ok()?;
    let statement = line[dot_pos + 1..].trim();

    // Try structured patterns first (most specific to least)
    let patterns: &[(&str, &str, fn(&str) -> Option<(&str, &str)>)] = &[];
    let _ = patterns; // suppress unused warning

    // Simple approach: split on common relationship phrases
    let relation_phrases = [
        (" is married to ", "married_to"),
        (" is a citizen of ", "citizen_of"),
        (" was born in the city of ", "born_in"),
        (" died in the city of ", "died_in"),
        (" plays the position of ", "plays_position"),
        (" was founded by ", "founded_by"),
        (" is famous for ", "famous_for"),
        (" was developed by ", "developed_by"),
        (" is located in the continent of ", "located_in_continent"),
        (" is located in the country of ", "located_in_country"),
        (" is associated with the sport of ", "sport_of"),
        (" was created by ", "created_by"),
        (" was created in the country of ", "created_in_country"),
        (" worked in the city of ", "worked_in"),
        (" was performed by ", "performed_by"),
        (" is employed by ", "employed_by"),
        (" speaks the language of ", "speaks_language"),
        (" is affiliated with the religion of ", "religion_of"),
        (" is a member of ", "member_of"),
        (" was written in the language of ", "written_in_language"),
        (" was founded in the city of ", "founded_in"),
    ];

    // Try direct subject-relation-object patterns
    for (phrase, rel_type) in &relation_phrases {
        if let Some(pos) = statement.find(phrase) {
            let subject = statement[..pos].trim().trim_end_matches('.');
            let object = statement[pos + phrase.len()..].trim().trim_end_matches('.');
            if !subject.is_empty() && !object.is_empty() {
                return Some(FactTriple {
                    index,
                    subject: subject.to_string(),
                    relation: rel_type.to_string(),
                    object: object.to_string(),
                    original: line.to_string(),
                });
            }
        }
    }

    // Try "The X of Y is Z" patterns
    let the_patterns = [
        ("The author of ", " is ", "author_of"),
        ("The capital of ", " is ", "capital_of"),
        ("The chairperson of ", " is ", "chairperson_of"),
        ("The chief executive officer of ", " is ", "ceo_of"),
        ("The director of ", " is ", "director_of"),
        (
            "The headquarters of ",
            " is located in the city of ",
            "headquarters_in",
        ),
        ("The official language of ", " is ", "official_language"),
        ("The type of music that ", " plays is ", "music_genre"),
        (
            "The name of the current head of the ",
            " government is ",
            "government_head",
        ),
        ("The univeristy where ", " was educated is ", "educated_at"),
        ("The company that produced ", " is ", "produced_by"),
        ("The genre of ", " is ", "genre_of"),
        ("The country of origin of ", " is ", "country_of_origin"),
    ];

    for (prefix, middle, rel_type) in &the_patterns {
        if let Some(rest) = statement.strip_prefix(prefix) {
            if let Some(mid_pos) = rest.find(middle) {
                let subject = rest[..mid_pos].trim();
                let object = rest[mid_pos + middle.len()..].trim().trim_end_matches('.');
                if !subject.is_empty() && !object.is_empty() {
                    return Some(FactTriple {
                        index,
                        subject: subject.to_string(),
                        relation: rel_type.to_string(),
                        object: object.to_string(),
                        original: line.to_string(),
                    });
                }
            }
        }
    }

    // Fallback: "X's child is Y"
    if let Some(pos) = statement.find("'s child is ") {
        let subject = statement[..pos].trim();
        let object = statement[pos + "'s child is ".len()..]
            .trim()
            .trim_end_matches('.');
        return Some(FactTriple {
            index,
            subject: subject.to_string(),
            relation: "child_of".to_string(),
            object: object.to_string(),
            original: line.to_string(),
        });
    }

    // Generic fallback: try to split on " is " or " was "
    for sep in [" is ", " was "] {
        if let Some(pos) = statement.find(sep) {
            let subject = statement[..pos].trim();
            let object = statement[pos + sep.len()..].trim().trim_end_matches('.');
            if !subject.is_empty() && !object.is_empty() && subject.len() > 2 {
                return Some(FactTriple {
                    index,
                    subject: subject.to_string(),
                    relation: "is_related".to_string(),
                    object: object.to_string(),
                    original: line.to_string(),
                });
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_basic_facts() {
        let text = "0. Thomas Kyd was born in the city of London.\n\
                     1. The chairperson of Fatah is Mahmoud Abbas.\n\
                     2. Amy Winehouse died in the city of Camden Town.";

        let (triples, _) = extract_facts(text);
        assert_eq!(triples.len(), 3);
        assert_eq!(triples[0].subject, "Thomas Kyd");
        assert_eq!(triples[0].relation, "born_in");
        assert_eq!(triples[0].object, "London");
        assert_eq!(triples[1].subject, "Fatah");
        assert_eq!(triples[1].relation, "chairperson_of");
        assert_eq!(triples[1].object, "Mahmoud Abbas");
    }

    #[test]
    fn detect_conflicts() {
        let text = "0. The chairperson of Fatah is Mahmoud Abbas.\n\
                     1. Some other fact.\n\
                     2. The chairperson of Fatah is Moshe Kahlon.";

        let (_, conflicts) = extract_facts(text);
        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0].old_fact.object, "Mahmoud Abbas");
        assert_eq!(conflicts[0].new_fact.object, "Moshe Kahlon");
        assert!(conflicts[0].new_fact.index > conflicts[0].old_fact.index);
    }

    #[test]
    fn no_conflict_same_value() {
        let text = "0. Amy Winehouse died in the city of Camden Town.\n\
                     1. Amy Winehouse died in the city of Camden Town.";

        let (_, conflicts) = extract_facts(text);
        assert_eq!(conflicts.len(), 0, "same value should not be a conflict");
    }
}

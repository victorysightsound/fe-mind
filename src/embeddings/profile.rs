pub const MINILM_MODEL_REPO: &str = "sentence-transformers/all-MiniLM-L6-v2";
pub const MINILM_SHORT_NAME: &str = "all-MiniLM-L6-v2";
pub const MINILM_CANONICAL_NAME: &str = "local-minilm";
pub const MINILM_DIMENSIONS: usize = 384;
pub const MINILM_PROFILE: &str =
    "local|sentence-transformers/all-MiniLM-L6-v2|384|v1|chars:none";

pub fn canonical_model_name(model: &str) -> String {
    if is_minilm_name(model) {
        MINILM_CANONICAL_NAME.to_string()
    } else {
        model.to_string()
    }
}

pub fn compatibility_model_names(model: &str) -> Vec<String> {
    if is_minilm_name(model) {
        vec![
            MINILM_CANONICAL_NAME.to_string(),
            MINILM_MODEL_REPO.to_string(),
            MINILM_SHORT_NAME.to_string(),
        ]
    } else {
        vec![model.to_string()]
    }
}

pub fn embedding_profile_for_model(model: &str, dimensions: usize) -> String {
    if is_minilm_name(model) && dimensions == MINILM_DIMENSIONS {
        MINILM_PROFILE.to_string()
    } else {
        format!("custom|{model}|{dimensions}")
    }
}

fn is_minilm_name(model: &str) -> bool {
    model == MINILM_MODEL_REPO || model == MINILM_SHORT_NAME || model == MINILM_CANONICAL_NAME
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minilm_names_normalize_to_repo() {
        assert_eq!(canonical_model_name(MINILM_MODEL_REPO), MINILM_CANONICAL_NAME);
        assert_eq!(canonical_model_name(MINILM_SHORT_NAME), MINILM_CANONICAL_NAME);
        assert_eq!(
            canonical_model_name(MINILM_CANONICAL_NAME),
            MINILM_CANONICAL_NAME
        );
    }

    #[test]
    fn minilm_names_expand_to_alias_set() {
        let names = compatibility_model_names(MINILM_CANONICAL_NAME);
        assert!(names.contains(&MINILM_CANONICAL_NAME.to_string()));
        assert!(names.contains(&MINILM_MODEL_REPO.to_string()));
        assert!(names.contains(&MINILM_SHORT_NAME.to_string()));
        assert!(names.contains(&MINILM_CANONICAL_NAME.to_string()));
    }
}

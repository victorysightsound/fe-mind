pub const RERANKER_CANONICAL_NAME: &str = "local-minilm-reranker";
pub const RERANKER_SHORT_NAME: &str = "minilm-reranker";
pub const RERANKER_MODEL_REPO: &str = "cross-encoder/ms-marco-MiniLM-L6-v2";
pub const RERANKER_PROFILE: &str =
    "local|cross-encoder/ms-marco-MiniLM-L6-v2|pair-score|v1|tokens:512";

pub fn canonical_reranker_name(model: &str) -> String {
    match model {
        RERANKER_CANONICAL_NAME | RERANKER_SHORT_NAME | RERANKER_MODEL_REPO => {
            RERANKER_CANONICAL_NAME.to_string()
        }
        other => other.to_string(),
    }
}

pub fn compatibility_reranker_names(model: &str) -> Vec<String> {
    let canonical = canonical_reranker_name(model);
    let mut names = vec![canonical];
    for alias in [
        RERANKER_CANONICAL_NAME,
        RERANKER_SHORT_NAME,
        RERANKER_MODEL_REPO,
    ] {
        if !names.iter().any(|name| name == alias) {
            names.push(alias.to_string());
        }
    }
    names
}

pub fn reranker_profile_for_model(model: &str) -> String {
    if compatibility_reranker_names(model)
        .iter()
        .any(|candidate| candidate == RERANKER_CANONICAL_NAME)
    {
        RERANKER_PROFILE.to_string()
    } else {
        model.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonicalizes_known_aliases() {
        assert_eq!(
            canonical_reranker_name(RERANKER_MODEL_REPO),
            RERANKER_CANONICAL_NAME
        );
        assert_eq!(
            canonical_reranker_name(RERANKER_SHORT_NAME),
            RERANKER_CANONICAL_NAME
        );
    }

    #[test]
    fn compatibility_names_include_repo() {
        let names = compatibility_reranker_names(RERANKER_CANONICAL_NAME);
        assert!(names.iter().any(|name| name == RERANKER_MODEL_REPO));
        assert!(names.iter().any(|name| name == RERANKER_SHORT_NAME));
    }
}

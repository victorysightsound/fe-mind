#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

#[cfg(all(feature = "api-embeddings", any(feature = "api-llm", feature = "cli-llm")))]
mod app {
    use std::env;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    use chrono::{DateTime, Utc};
    use femind::embeddings::{ApiBackend, EmbeddingBackend};
    use femind::engine::{EngineConfig, MemoryEngine, VectorSearchMode};
    #[cfg(feature = "api-llm")]
    use femind::llm::ApiLlmCallback;
    #[cfg(feature = "cli-llm")]
    use femind::llm::CliLlmCallback;
    use femind::search::SearchMode;
    use femind::traits::{LlmCallback, MemoryRecord, MemoryType};
    use serde::{Deserialize, Serialize};

    const DEFAULT_SCENARIOS: &str = "eval/practical/scenarios.json";
    const DEFAULT_BASE_URL: &str = "https://api.deepinfra.com/v1/openai";
    const DEFAULT_KEY_CMD: &str = "op read 'op://Personal/Deep Infra/credential' 2>/dev/null";
    const DEFAULT_EMBED_MODEL: &str = "sentence-transformers/all-MiniLM-L6-v2";
    const DEFAULT_API_EXTRACT_MODEL: &str = "openai/gpt-oss-120b";
    const DEFAULT_CODEX_EXTRACT_MODEL: &str = "gpt-5.4-mini";

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct EvalMemory {
        id: Option<i64>,
        text: String,
        source: String,
        memory_type: MemoryType,
        created_at: DateTime<Utc>,
    }

    impl MemoryRecord for EvalMemory {
        fn id(&self) -> Option<i64> {
            self.id
        }

        fn searchable_text(&self) -> String {
            self.text.clone()
        }

        fn memory_type(&self) -> MemoryType {
            self.memory_type
        }

        fn importance(&self) -> u8 {
            7
        }

        fn created_at(&self) -> DateTime<Utc> {
            self.created_at
        }

        fn category(&self) -> Option<&str> {
            Some(&self.source)
        }
    }

    #[derive(Debug, Deserialize)]
    struct ScenarioRecord {
        timestamp: String,
        source: String,
        memory_type: String,
        text: String,
    }

    #[derive(Debug, Deserialize)]
    struct RetrievalCheck {
        query: String,
        expected_answer: String,
    }

    #[derive(Debug, Deserialize)]
    struct ExtractionCheck {
        expected_fact: String,
    }

    #[derive(Debug, Deserialize)]
    struct AbstentionCheck {
        query: String,
        expected_behavior: String,
    }

    #[derive(Debug, Deserialize)]
    struct Scenario {
        id: String,
        title: String,
        category: String,
        goal: String,
        records: Vec<ScenarioRecord>,
        #[serde(default)]
        retrieval_checks: Vec<RetrievalCheck>,
        #[serde(default)]
        extraction_checks: Vec<ExtractionCheck>,
        #[serde(default)]
        abstention_checks: Vec<AbstentionCheck>,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum EvalMode {
        Retrieval,
        Extraction,
        All,
    }

    impl EvalMode {
        fn from_str(value: &str) -> Result<Self, String> {
            match value {
                "retrieval" => Ok(Self::Retrieval),
                "extraction" => Ok(Self::Extraction),
                "all" => Ok(Self::All),
                other => Err(format!(
                    "unknown mode '{other}', expected retrieval | extraction | all"
                )),
            }
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum ExtractionBackend {
        Api,
        CodexCli,
    }

    impl ExtractionBackend {
        fn from_str(value: &str) -> Result<Self, String> {
            match value {
                "api" => Ok(Self::Api),
                "codex-cli" => Ok(Self::CodexCli),
                other => Err(format!(
                    "unknown extract backend '{other}', expected api | codex-cli"
                )),
            }
        }

        fn default_model(self) -> &'static str {
            match self {
                Self::Api => DEFAULT_API_EXTRACT_MODEL,
                Self::CodexCli => DEFAULT_CODEX_EXTRACT_MODEL,
            }
        }

        fn name(self) -> &'static str {
            match self {
                Self::Api => "api",
                Self::CodexCli => "codex-cli",
            }
        }
    }

    #[derive(Debug)]
    struct Config {
        scenarios_path: PathBuf,
        db_path: Option<PathBuf>,
        summary_path: Option<PathBuf>,
        mode: EvalMode,
        vector_mode: VectorSearchMode,
        top_k: usize,
        base_url: String,
        api_key_env: String,
        key_cmd: String,
        embedding_model: String,
        extraction_backend: ExtractionBackend,
        extraction_model: String,
    }

    impl Config {
        fn from_args() -> Result<Self, String> {
            let mut scenarios_path = PathBuf::from(DEFAULT_SCENARIOS);
            let mut db_path = None;
            let mut summary_path = None;
            let mut mode = EvalMode::All;
            let mut vector_mode = VectorSearchMode::Exact;
            let mut top_k = 3usize;
            let mut base_url = DEFAULT_BASE_URL.to_string();
            let mut api_key_env =
                env::var("FEMIND_API_KEY_ENV").unwrap_or_else(|_| "FEMIND_API_KEY".to_string());
            let mut key_cmd = env::var("FEMIND_DEEPINFRA_KEY_CMD")
                .unwrap_or_else(|_| DEFAULT_KEY_CMD.to_string());
            let mut embedding_model = env::var("FEMIND_EMBED_MODEL")
                .unwrap_or_else(|_| DEFAULT_EMBED_MODEL.to_string());
            let extraction_backend = ExtractionBackend::from_str(
                &env::var("FEMIND_EXTRACT_BACKEND").unwrap_or_else(|_| "api".to_string()),
            )?;
            let mut extraction_backend = extraction_backend;
            let mut extraction_model = env::var("FEMIND_EXTRACT_MODEL")
                .unwrap_or_else(|_| extraction_backend.default_model().to_string());

            let mut args = env::args().skip(1);
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--scenarios" => {
                        scenarios_path = PathBuf::from(
                            args.next().ok_or("--scenarios requires a path")?,
                        );
                    }
                    "--db" => {
                        db_path = Some(PathBuf::from(args.next().ok_or("--db requires a path")?));
                    }
                    "--summary" => {
                        summary_path =
                            Some(PathBuf::from(args.next().ok_or("--summary requires a path")?));
                    }
                    "--mode" => {
                        mode = EvalMode::from_str(&args.next().ok_or("--mode requires a value")?)?;
                    }
                    "--vector-mode" => {
                        vector_mode = match args
                            .next()
                            .ok_or("--vector-mode requires a value")?
                            .as_str()
                        {
                            "off" => VectorSearchMode::Off,
                            "exact" => VectorSearchMode::Exact,
                            "ann" => VectorSearchMode::Ann,
                            other => {
                                return Err(format!(
                                    "unknown vector mode '{other}', expected off | exact | ann"
                                ))
                            }
                        };
                    }
                    "--top-k" => {
                        top_k = args
                            .next()
                            .ok_or("--top-k requires a value")?
                            .parse()
                            .map_err(|_| "--top-k must be an integer".to_string())?;
                    }
                    "--base-url" => {
                        base_url = args.next().ok_or("--base-url requires a value")?;
                    }
                    "--api-key-env" => {
                        api_key_env = args.next().ok_or("--api-key-env requires a value")?;
                    }
                    "--key-cmd" => {
                        key_cmd = args.next().ok_or("--key-cmd requires a value")?;
                    }
                    "--embedding-model" => {
                        embedding_model =
                            args.next().ok_or("--embedding-model requires a value")?;
                    }
                    "--extract-backend" => {
                        extraction_backend = ExtractionBackend::from_str(
                            &args.next().ok_or("--extract-backend requires a value")?,
                        )?;
                        if env::var_os("FEMIND_EXTRACT_MODEL").is_none() {
                            extraction_model = extraction_backend.default_model().to_string();
                        }
                    }
                    "--extraction-model" => {
                        extraction_model =
                            args.next().ok_or("--extraction-model requires a value")?;
                    }
                    "--help" | "-h" => {
                        print_help();
                        std::process::exit(0);
                    }
                    other => return Err(format!("unknown argument: {other}")),
                }
            }

            Ok(Self {
                scenarios_path,
                db_path,
                summary_path,
                mode,
                vector_mode,
                top_k,
                base_url,
                api_key_env,
                key_cmd,
                embedding_model,
                extraction_backend,
                extraction_model,
            })
        }
    }

    #[derive(Debug, Serialize)]
    struct ObservedHit {
        text: String,
        score: f32,
    }

    #[derive(Debug, Serialize)]
    struct CheckReport {
        query: String,
        passed: bool,
        expected: String,
        observed: Vec<ObservedHit>,
    }

    #[derive(Debug, Serialize)]
    struct ScenarioReport {
        id: String,
        title: String,
        category: String,
        mode: String,
        retrieval: Vec<CheckReport>,
        extraction: Vec<CheckReport>,
        abstention: Vec<CheckReport>,
    }

    #[derive(Debug, Serialize)]
    struct RunMetadata {
        generated_at: DateTime<Utc>,
        scenarios_path: String,
        scenario_count: usize,
        mode: String,
        vector_mode: String,
        top_k: usize,
        embedding_model: String,
        extract_backend: String,
        extraction_model: String,
        duration_ms: u128,
    }

    #[derive(Debug, Serialize)]
    struct RunSummary {
        metadata: RunMetadata,
        total_checks: usize,
        passed_checks: usize,
        pass_rate: f32,
        reports: Vec<ScenarioReport>,
    }

    pub fn run() -> Result<(), Box<dyn std::error::Error>> {
        let config = Config::from_args().map_err(|e| format!("argument error: {e}"))?;
        let scenarios = load_scenarios(&config.scenarios_path)?;
        let started_at = Instant::now();

        let api_key = resolve_api_key(&config)?;
        let embedder =
            ApiBackend::new(&config.base_url, api_key.clone(), &config.embedding_model, 384);
        let extractor = build_extractor(&config, api_key);

        println!("femind practical evaluation");
        println!("==========================");
        println!("scenarios: {}", scenarios.len());
        println!("mode: {}", mode_name(config.mode));
        println!("vector_mode: {}", config.vector_mode);
        println!("embedding_model: {}", embedder.model_name());
        println!("extract_backend: {}", config.extraction_backend.name());
        println!("extraction_model: {}", extractor.model_name());
        println!();

        let mut reports = Vec::new();
        let mut total_checks = 0usize;
        let mut passed_checks = 0usize;

        for scenario in &scenarios {
            println!("[{}] {} ({})", scenario.id, scenario.title, scenario.category);
            println!("goal: {}", scenario.goal);

            let scenario_db = scenario_db_path(&config, &scenario.id)?;
            let report = run_scenario(scenario, &scenario_db, &config, extractor.as_ref())?;

            let scenario_passed = report
                .retrieval
                .iter()
                .chain(report.extraction.iter())
                .chain(report.abstention.iter())
                .filter(|c| c.passed)
                .count();
            let scenario_total = report.retrieval.len() + report.extraction.len() + report.abstention.len();
            passed_checks += scenario_passed;
            total_checks += scenario_total;

            println!("checks: {scenario_passed}/{scenario_total} passed");
            println!();

            reports.push(report);
        }

        println!("summary: {passed_checks}/{total_checks} checks passed");

        let duration_ms = started_at.elapsed().as_millis();
        let summary = RunSummary {
            metadata: RunMetadata {
                generated_at: Utc::now(),
                scenarios_path: config.scenarios_path.display().to_string(),
                scenario_count: scenarios.len(),
                mode: mode_name(config.mode).to_string(),
                vector_mode: config.vector_mode.to_string(),
                top_k: config.top_k,
                embedding_model: embedder.model_name().to_string(),
                extract_backend: config.extraction_backend.name().to_string(),
                extraction_model: extractor.model_name().to_string(),
                duration_ms,
            },
            total_checks,
            passed_checks,
            pass_rate: if total_checks == 0 {
                0.0
            } else {
                passed_checks as f32 / total_checks as f32
            },
            reports,
        };

        if let Some(path) = config.summary_path {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(&path, serde_json::to_vec_pretty(&summary)?)?;
            println!("summary_file: {}", path.display());
        }

        Ok(())
    }

    fn run_scenario(
        scenario: &Scenario,
        db_path: &Path,
        config: &Config,
        extractor: &dyn LlmCallback,
    ) -> Result<ScenarioReport, Box<dyn std::error::Error>> {
        if db_path.exists() {
            fs::remove_file(db_path)?;
        }

        let mut engine = MemoryEngine::<EvalMemory>::builder()
            .database(db_path.to_string_lossy().into_owned())
            .embedding_backend(ApiBackend::new(
                &config.base_url,
                resolve_api_key(config)?,
                &config.embedding_model,
                384,
            ))
            .build()?;
        engine.config = EngineConfig {
            embedding_enabled: !matches!(config.vector_mode, VectorSearchMode::Off),
            vector_search_mode: config.vector_mode,
            ..EngineConfig::default()
        };

        if matches!(config.mode, EvalMode::Retrieval | EvalMode::All) {
            let records: Vec<EvalMemory> = scenario
                .records
                .iter()
                .map(to_eval_memory)
                .collect::<Result<Vec<_>, _>>()?;
            let _ = engine.store_batch(&records)?;
        }

        let mut retrieval = Vec::new();
        if matches!(config.mode, EvalMode::Retrieval | EvalMode::All) {
            for check in &scenario.retrieval_checks {
                let observed = top_hits(&engine, &check.query, config.top_k)?;
                let passed = observed
                    .iter()
                    .any(|hit| expected_match(&hit.text, &check.query, &check.expected_answer));
                retrieval.push(CheckReport {
                    query: check.query.clone(),
                    passed,
                    expected: check.expected_answer.clone(),
                    observed,
                });
            }
        }

        let mut extraction = Vec::new();
        if matches!(config.mode, EvalMode::Extraction | EvalMode::All)
            && !scenario.extraction_checks.is_empty()
        {
            let raw_text = scenario
                .records
                .iter()
                .map(|r| format!("[{}] {}", r.source, r.text))
                .collect::<Vec<_>>()
                .join("\n");
            let _ = engine.store_with_extraction(&raw_text, extractor)?;
            let extracted_texts = all_memory_texts(&engine)?;

            for check in &scenario.extraction_checks {
                let observed = extracted_texts.clone();
                let passed = observed
                    .iter()
                    .any(|hit| expected_match(hit, &check.expected_fact, &check.expected_fact));
                extraction.push(CheckReport {
                    query: check.expected_fact.clone(),
                    passed,
                    expected: check.expected_fact.clone(),
                    observed: observed
                        .into_iter()
                        .map(|text| ObservedHit { text, score: 1.0 })
                        .collect(),
                });
            }
        }

        let mut abstention = Vec::new();
        if matches!(config.mode, EvalMode::Retrieval | EvalMode::All) {
            for check in &scenario.abstention_checks {
                let observed = top_hits(&engine, &check.query, config.top_k)?;
                let passed = check.expected_behavior == "abstain" && observed.is_empty();
                abstention.push(CheckReport {
                    query: check.query.clone(),
                    passed,
                    expected: check.expected_behavior.clone(),
                    observed,
                });
            }
        }

        Ok(ScenarioReport {
            id: scenario.id.clone(),
            title: scenario.title.clone(),
            category: scenario.category.clone(),
            mode: mode_name(config.mode).to_string(),
            retrieval,
            extraction,
            abstention,
        })
    }

    fn build_extractor(config: &Config, api_key: String) -> Box<dyn LlmCallback> {
        match config.extraction_backend {
            ExtractionBackend::Api => {
                #[cfg(feature = "api-llm")]
                {
                    Box::new(ApiLlmCallback::new(
                        &config.base_url,
                        api_key,
                        &config.extraction_model,
                    ))
                }
                #[cfg(not(feature = "api-llm"))]
                {
                    let _ = api_key;
                    panic!("api extraction backend requires api-llm feature");
                }
            }
            ExtractionBackend::CodexCli => {
                #[cfg(feature = "cli-llm")]
                {
                    let _ = api_key;
                    Box::new(CliLlmCallback::codex(&config.extraction_model))
                }
                #[cfg(not(feature = "cli-llm"))]
                {
                    let _ = api_key;
                    panic!("codex-cli extraction backend requires cli-llm feature");
                }
            }
        }
    }

    fn top_hits(
        engine: &MemoryEngine<EvalMemory>,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<ObservedHit>, Box<dyn std::error::Error>> {
        let results = engine
            .search(query)
            .mode(SearchMode::Auto)
            .limit(top_k)
            .execute()?;

        let mut hits = Vec::new();
        for result in results {
            let text = engine.database().with_reader(|conn| {
                conn.query_row(
                    "SELECT searchable_text FROM memories WHERE id = ?1",
                    [result.memory_id],
                    |row| row.get::<_, String>(0),
                )
                .map_err(Into::into)
            })?;
            hits.push(ObservedHit {
                text,
                score: result.score,
            });
        }
        Ok(hits)
    }

    fn all_memory_texts(
        engine: &MemoryEngine<EvalMemory>,
    ) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        Ok(engine.database().with_reader(|conn| {
            let mut stmt = conn.prepare("SELECT searchable_text FROM memories ORDER BY id")?;
            let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
            let mut texts = Vec::new();
            for row in rows {
                texts.push(row?);
            }
            Ok(texts)
        })?)
    }

    fn to_eval_memory(record: &ScenarioRecord) -> Result<EvalMemory, Box<dyn std::error::Error>> {
        Ok(EvalMemory {
            id: None,
            text: record.text.clone(),
            source: record.source.clone(),
            memory_type: parse_memory_type(&record.memory_type)?,
            created_at: DateTime::parse_from_rfc3339(&record.timestamp)?.with_timezone(&Utc),
        })
    }

    fn parse_memory_type(value: &str) -> Result<MemoryType, Box<dyn std::error::Error>> {
        MemoryType::from_str(&value.to_lowercase())
            .ok_or_else(|| format!("unknown memory_type '{value}'").into())
    }

    fn load_scenarios(path: &Path) -> Result<Vec<Scenario>, Box<dyn std::error::Error>> {
        let raw = fs::read_to_string(path)?;
        Ok(serde_json::from_str(&raw)?)
    }

    fn resolve_api_key(config: &Config) -> Result<String, Box<dyn std::error::Error>> {
        if let Ok(value) = env::var(&config.api_key_env) {
            let trimmed = value.trim();
            if !trimmed.is_empty() {
                return Ok(trimmed.to_string());
            }
        }

        let output = std::process::Command::new("sh")
            .args(["-c", &config.key_cmd])
            .output()?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("key_cmd error: {stderr}").into());
        }

        let api_key = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if api_key.is_empty() {
            return Err("key_cmd returned empty key".into());
        }
        Ok(api_key)
    }

    fn scenario_db_path(config: &Config, scenario_id: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
        if let Some(base) = &config.db_path {
            if base.extension().is_some() {
                let stem = base.file_stem().and_then(|s| s.to_str()).unwrap_or("practical-eval");
                let parent = base.parent().unwrap_or_else(|| Path::new("."));
                fs::create_dir_all(parent)?;
                return Ok(parent.join(format!("{stem}-{scenario_id}.db")));
            }
            fs::create_dir_all(base)?;
            return Ok(base.join(format!("{scenario_id}.db")));
        }

        let root = PathBuf::from("target/practical-eval");
        fs::create_dir_all(&root)?;
        Ok(root.join(format!("{scenario_id}.db")))
    }

    fn expected_match(observed: &str, query: &str, expected: &str) -> bool {
        let observed_normalized = normalize(observed);
        let expected_normalized = normalize(expected);
        if observed_normalized.contains(&expected_normalized) {
            return true;
        }

        if matches_yes_no_state_answer(&observed_normalized, &normalize(query), &expected_normalized) {
            return true;
        }

        let observed_tokens = meaning_tokens(observed);
        let expected_tokens = meaning_tokens(expected);
        if expected_tokens.is_empty() {
            return false;
        }

        let overlap = expected_tokens
            .iter()
            .filter(|token| observed_tokens.contains(*token))
            .count();
        let recall = overlap as f32 / expected_tokens.len() as f32;

        let min_overlap = if expected_tokens.len() <= 2 { expected_tokens.len() } else { 2 };
        overlap >= min_overlap && recall >= 0.6
    }

    fn matches_yes_no_state_answer(observed: &str, query: &str, expected: &str) -> bool {
        let negative_expected = expected.starts_with("no ")
            || expected == "no"
            || expected.contains("not active")
            || expected.contains("superseded");
        let asks_current_state = query.contains("still")
            || query.contains("active")
            || query.contains("current")
            || query.starts_with("is ");

        if !(negative_expected && asks_current_state) {
            return false;
        }

        observed.contains("superseded")
            || observed.contains("no longer")
            || observed.contains("not active")
            || observed.contains("prior ")
            || observed.contains("former ")
            || observed.contains("outdated")
    }

    fn normalize(value: &str) -> String {
        value
            .to_lowercase()
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() || c.is_ascii_whitespace() { c } else { ' ' })
            .collect::<String>()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn meaning_tokens(value: &str) -> std::collections::BTreeSet<String> {
        normalize(value)
            .split_whitespace()
            .filter_map(canonical_token)
            .collect()
    }

    fn canonical_token(token: &str) -> Option<String> {
        let token = match token {
            "the" | "a" | "an" | "is" | "are" | "was" | "were" | "be" | "been" | "being"
            | "to" | "for" | "of" | "in" | "on" | "at" | "by" | "and" | "or" | "that"
            | "this" | "it" | "its" | "still" | "then" | "than" | "because" | "what"
            | "which" | "who" | "should" | "not" | "do" | "does" | "did" | "yet"
            | "after" | "before" | "over" | "under" | "with" | "without" | "from"
            | "into" | "about" | "no" | "current" | "earlier" => return None,
            "keep" | "used" | "use" => "prefer",
            "tried" | "try" => "first",
            "improved" | "good" | "looked" => "better",
            "happen" | "performed" => "run",
            "built" | "build" => "build",
            "preferred" | "prefer" => "prefer",
            "superseded" => "superseded",
            other => other,
        };

        let stemmed = token
            .trim_end_matches("ing")
            .trim_end_matches("ed")
            .trim_end_matches('s');
        if stemmed.is_empty() {
            None
        } else {
            Some(stemmed.to_string())
        }
    }

    fn mode_name(mode: EvalMode) -> &'static str {
        match mode {
            EvalMode::Retrieval => "retrieval",
            EvalMode::Extraction => "extraction",
            EvalMode::All => "all",
        }
    }

    fn print_help() {
        println!(
            "Usage: cargo run --example practical_eval --features api-embeddings,api-llm,ann -- [options]\n\
             \n\
             Options:\n\
             \t--scenarios <path>         Path to scenarios JSON (default: eval/practical/scenarios.json)\n\
             \t--db <path>                Output DB file or directory for per-scenario runs\n\
             \t--summary <path>           Write JSON summary to a file\n\
             \t--mode <retrieval|extraction|all>\n\
             \t--vector-mode <off|exact|ann>\n\
             \t--top-k <n>                Number of retrieved records to inspect (default: 3)\n\
             \t--base-url <url>           OpenAI-compatible API base URL\n\
             \t--api-key-env <name>       Environment variable to read before using --key-cmd\n\
             \t--key-cmd <cmd>            Shell command that prints the API key\n\
             \t--embedding-model <model>  Embedding model name\n\
             \t--extract-backend <kind>   Extraction backend: api | codex-cli\n\
             \t--extraction-model <model> Extraction model name\n"
        );
    }

    #[cfg(test)]
    mod tests {
        use super::{expected_match, matches_yes_no_state_answer};

        #[test]
        fn matcher_accepts_superseded_yes_no_answer() {
            let observed =
                "The prior desktop-first idea is superseded. Current build order is femind first, then feloop.";
            let query = "Is desktop-first still the active plan?";
            let expected = "No. That plan was superseded.";

            assert!(matches_yes_no_state_answer(
                &observed.to_lowercase(),
                &query.to_lowercase(),
                &expected.to_lowercase()
            ));
            assert!(expected_match(observed, query, expected));
        }

        #[test]
        fn matcher_rejects_positive_answer_for_negative_state_check() {
            let observed = "Desktop-first is still the active plan for the next release.";
            let query = "Is desktop-first still the active plan?";
            let expected = "No. That plan was superseded.";

            assert!(!expected_match(observed, query, expected));
        }
    }
}

#[cfg(all(feature = "api-embeddings", any(feature = "api-llm", feature = "cli-llm")))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    app::run()
}

#[cfg(not(all(
    feature = "api-embeddings",
    any(feature = "api-llm", feature = "cli-llm")
)))]
fn main() {
    eprintln!(
        "This example requires features: api-embeddings and one extraction backend (api-llm or cli-llm).\n\
         Example: cargo run --example practical_eval --features api-embeddings,cli-llm,ann -- --help"
    );
    std::process::exit(1);
}

#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

#[cfg(all(feature = "api-embeddings", feature = "api-llm"))]
mod app {
    use std::env;
    use std::fs;
    use std::path::{Path, PathBuf};

    use chrono::{DateTime, Utc};
    use femind::embeddings::{ApiBackend, EmbeddingBackend};
    use femind::engine::{EngineConfig, MemoryEngine, VectorSearchMode};
    use femind::llm::ApiLlmCallback;
    use femind::search::SearchMode;
    use femind::traits::{LlmCallback, MemoryRecord, MemoryType};
    use serde::{Deserialize, Serialize};

    const DEFAULT_SCENARIOS: &str = "eval/practical/scenarios.json";
    const DEFAULT_BASE_URL: &str = "https://api.deepinfra.com/v1/openai";
    const DEFAULT_KEY_CMD: &str = "op read 'op://Personal/Deep Infra/credential' 2>/dev/null";
    const DEFAULT_EMBED_MODEL: &str = "sentence-transformers/all-MiniLM-L6-v2";
    const DEFAULT_EXTRACT_MODEL: &str = "openai/gpt-oss-120b";

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct EvalMemory {
        id: Option<i64>,
        text: String,
        source: String,
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
            MemoryType::Semantic
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

    #[derive(Debug)]
    struct Config {
        scenarios_path: PathBuf,
        db_path: Option<PathBuf>,
        summary_path: Option<PathBuf>,
        mode: EvalMode,
        vector_mode: VectorSearchMode,
        top_k: usize,
        base_url: String,
        key_cmd: String,
        embedding_model: String,
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
            let mut key_cmd = env::var("FEMIND_DEEPINFRA_KEY_CMD")
                .unwrap_or_else(|_| DEFAULT_KEY_CMD.to_string());
            let mut embedding_model = env::var("FEMIND_EMBED_MODEL")
                .unwrap_or_else(|_| DEFAULT_EMBED_MODEL.to_string());
            let mut extraction_model = env::var("FEMIND_EXTRACT_MODEL")
                .unwrap_or_else(|_| DEFAULT_EXTRACT_MODEL.to_string());

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
                    "--key-cmd" => {
                        key_cmd = args.next().ok_or("--key-cmd requires a value")?;
                    }
                    "--embedding-model" => {
                        embedding_model =
                            args.next().ok_or("--embedding-model requires a value")?;
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
                key_cmd,
                embedding_model,
                extraction_model,
            })
        }
    }

    #[derive(Debug, Serialize)]
    struct CheckReport {
        query: String,
        passed: bool,
        expected: String,
        observed: Vec<String>,
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

    pub fn run() -> Result<(), Box<dyn std::error::Error>> {
        let config = Config::from_args().map_err(|e| format!("argument error: {e}"))?;
        let scenarios = load_scenarios(&config.scenarios_path)?;

        let embedder = ApiBackend::with_key_cmd(
            &config.base_url,
            &config.key_cmd,
            &config.embedding_model,
            384,
        )?;

        let extractor =
            ApiLlmCallback::with_key_cmd(&config.base_url, &config.key_cmd, &config.extraction_model)?;

        println!("femind practical evaluation");
        println!("==========================");
        println!("scenarios: {}", scenarios.len());
        println!("mode: {}", mode_name(config.mode));
        println!("vector_mode: {}", config.vector_mode);
        println!("embedding_model: {}", embedder.model_name());
        println!("extraction_model: {}", extractor.model_name());
        println!();

        let mut reports = Vec::new();
        let mut total_checks = 0usize;
        let mut passed_checks = 0usize;

        for scenario in &scenarios {
            println!("[{}] {} ({})", scenario.id, scenario.title, scenario.category);
            println!("goal: {}", scenario.goal);

            let scenario_db = scenario_db_path(&config, &scenario.id)?;
            let report = run_scenario(scenario, &scenario_db, &config, &extractor)?;

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

        if let Some(path) = config.summary_path {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(&path, serde_json::to_vec_pretty(&reports)?)?;
            println!("summary_file: {}", path.display());
        }

        Ok(())
    }

    fn run_scenario(
        scenario: &Scenario,
        db_path: &Path,
        config: &Config,
        extractor: &ApiLlmCallback,
    ) -> Result<ScenarioReport, Box<dyn std::error::Error>> {
        if db_path.exists() {
            fs::remove_file(db_path)?;
        }

        let mut engine = MemoryEngine::<EvalMemory>::builder()
            .database(db_path.to_string_lossy().into_owned())
            .embedding_backend(ApiBackend::with_key_cmd(
                &config.base_url,
                &config.key_cmd,
                &config.embedding_model,
                384,
            )?)
            .build()?;
        engine.config = EngineConfig {
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
                    .any(|hit| contains_normalized(hit, &check.expected_answer));
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

            for check in &scenario.extraction_checks {
                let observed = top_hits(&engine, &check.expected_fact, config.top_k)?;
                let passed = observed
                    .iter()
                    .any(|hit| contains_normalized(hit, &check.expected_fact));
                extraction.push(CheckReport {
                    query: check.expected_fact.clone(),
                    passed,
                    expected: check.expected_fact.clone(),
                    observed,
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

    fn top_hits(
        engine: &MemoryEngine<EvalMemory>,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        let results = engine
            .search(query)
            .mode(SearchMode::Auto)
            .limit(top_k)
            .execute()?;

        let mut hits = Vec::new();
        for result in results {
            if let Some(record) = engine.get(result.memory_id)? {
                hits.push(record.text);
            }
        }
        Ok(hits)
    }

    fn to_eval_memory(record: &ScenarioRecord) -> Result<EvalMemory, Box<dyn std::error::Error>> {
        Ok(EvalMemory {
            id: None,
            text: record.text.clone(),
            source: record.source.clone(),
            created_at: DateTime::parse_from_rfc3339(&record.timestamp)?.with_timezone(&Utc),
        })
    }

    fn load_scenarios(path: &Path) -> Result<Vec<Scenario>, Box<dyn std::error::Error>> {
        let raw = fs::read_to_string(path)?;
        Ok(serde_json::from_str(&raw)?)
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

    fn contains_normalized(haystack: &str, needle: &str) -> bool {
        normalize(haystack).contains(&normalize(needle))
    }

    fn normalize(value: &str) -> String {
        value
            .to_lowercase()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
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
             \t--key-cmd <cmd>            Shell command that prints the API key\n\
             \t--embedding-model <model>  Embedding model name\n\
             \t--extraction-model <model> Extraction model name\n"
        );
    }
}

#[cfg(all(feature = "api-embeddings", feature = "api-llm"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    app::run()
}

#[cfg(not(all(feature = "api-embeddings", feature = "api-llm")))]
fn main() {
    eprintln!(
        "This example requires features: api-embeddings and api-llm.\n\
         Example: cargo run --example practical_eval --features api-embeddings,api-llm,ann -- --help"
    );
    std::process::exit(1);
}

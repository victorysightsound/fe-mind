mod dataset;
mod download;
mod ingest;
mod judge;
mod llm;
mod metrics;
mod retrieval;
mod verify;

use std::collections::HashSet;
use std::path::Path;

use anyhow::Result;
use clap::{Parser, Subcommand};
use dataset::{DatasetVariant, EvalResult};
use indicatif::{ProgressBar, ProgressStyle};

#[derive(Parser)]
#[command(name = "mindcore-bench")]
#[command(about = "LongMemEval benchmark harness for MindCore")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Download the LongMemEval dataset
    Download {
        /// Dataset variant: oracle, small, medium
        #[arg(short, long, default_value = "oracle")]
        variant: String,
    },
    /// Run the benchmark
    Run {
        /// Dataset variant: oracle, small, medium
        #[arg(short, long, default_value = "oracle")]
        variant: String,
        /// Number of questions to evaluate (0 = all)
        #[arg(short, long, default_value = "0")]
        limit: usize,
        /// Output file for results (appends; skips already-evaluated questions)
        #[arg(short, long, default_value = "results.jsonl")]
        output: String,
        /// Context budget in tokens (0 = unlimited)
        #[arg(short, long, default_value = "0")]
        budget: usize,
        /// Model for answer generation (default: sonnet)
        #[arg(short, long, default_value = "sonnet")]
        model: String,
        /// Model for judging (default: sonnet)
        #[arg(long, default_value = "sonnet")]
        judge_model: String,
        /// Disable self-verification pass
        #[arg(long)]
        no_verify: bool,
    },
    /// Show metrics from a results file
    Report {
        /// Path to results JSONL file
        #[arg(default_value = "results.jsonl")]
        results_file: String,
    },
    /// Show dataset statistics
    Stats {
        /// Dataset variant: oracle, small, medium
        #[arg(short, long, default_value = "oracle")]
        variant: String,
    },
}

fn parse_variant(s: &str) -> DatasetVariant {
    match s.to_lowercase().as_str() {
        "oracle" | "o" => DatasetVariant::Oracle,
        "small" | "s" => DatasetVariant::Small,
        "medium" | "m" => DatasetVariant::Medium,
        _ => {
            eprintln!("Unknown variant '{s}', using oracle");
            DatasetVariant::Oracle
        }
    }
}

/// Load already-completed question IDs from a JSONL results file.
fn load_completed_ids(path: &str) -> HashSet<String> {
    let mut ids = HashSet::new();
    if let Ok(data) = std::fs::read_to_string(path) {
        for line in data.lines() {
            if let Ok(result) = serde_json::from_str::<EvalResult>(line) {
                ids.insert(result.question_id);
            }
        }
    }
    ids
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Download { variant } => {
            let v = parse_variant(&variant);
            let path = download::download_dataset(v).await?;
            println!("Dataset ready: {}", path.display());
        }
        Commands::Stats { variant } => {
            let v = parse_variant(&variant);
            let path = download::download_dataset(v).await?;
            let ds = dataset::Dataset::load(&path)?;

            println!("LongMemEval Dataset Statistics");
            println!("==============================");
            println!("Variant: {}", v.filename());
            println!("Questions: {}", ds.questions.len());
            println!("Abstention: {}", ds.abstention_count());
            println!();

            let counts = ds.count_by_type();
            println!("By question type:");
            for (qt, count) in &counts {
                println!("  {}: {count}", qt.display_name());
            }

            let total_turns: usize = ds.questions.iter().map(|q| q.total_turns()).sum();
            let total_sessions: usize =
                ds.questions.iter().map(|q| q.haystack_sessions.len()).sum();
            println!();
            println!("Total sessions: {total_sessions}");
            println!("Total turns: {total_turns}");
        }
        Commands::Run {
            variant,
            limit,
            output,
            budget,
            model,
            judge_model,
            no_verify,
        } => {
            let v = parse_variant(&variant);
            let path = download::download_dataset(v).await?;
            let ds = dataset::Dataset::load(&path)?;

            let gen_client = llm::ClaudeCliClient::with_model(&model);
            let judge_client = llm::ClaudeCliClient::with_model(&judge_model);

            let questions = if limit > 0 {
                &ds.questions[..limit.min(ds.questions.len())]
            } else {
                &ds.questions
            };

            // Resume support: skip already-evaluated questions
            let completed = load_completed_ids(&output);
            let remaining: Vec<_> = questions
                .iter()
                .filter(|q| !completed.contains(&q.question_id))
                .collect();

            if !completed.is_empty() {
                println!(
                    "Resuming: {}/{} already completed, {} remaining",
                    completed.len(),
                    questions.len(),
                    remaining.len()
                );
            }

            println!(
                "Running LongMemEval benchmark: {} questions, budget={budget} tokens",
                remaining.len()
            );
            let verify_status = if no_verify { "off" } else { "on (multi-session, temporal, knowledge-update)" };
            println!("Generation: {model} | Judge: {judge_model} | Verify: {verify_status}");
            println!("Using Claude Code CLI (subscription, no API tokens)\n");

            if remaining.is_empty() {
                println!("All questions already evaluated. Loading results for report.\n");
                let data = std::fs::read_to_string(&output)?;
                let results: Vec<EvalResult> = data
                    .lines()
                    .filter(|l| !l.trim().is_empty())
                    .map(|l| serde_json::from_str(l))
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                let m = metrics::compute_metrics(&results);
                metrics::print_report(&m);
                return Ok(());
            }

            let pb = ProgressBar::new(remaining.len() as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("[{bar:40}] {pos}/{len} ({eta}) {msg}")
                    .expect("style")
                    .progress_chars("=> "),
            );

            // Ensure output directory exists
            if let Some(parent) = Path::new(&output).parent() {
                if !parent.as_os_str().is_empty() {
                    std::fs::create_dir_all(parent)?;
                }
            }

            let mut correct_count = 0_usize;
            let mut error_count = 0_usize;
            let total_done = completed.len();

            for (i, question) in remaining.iter().enumerate() {
                pb.set_message(format!("{}", question.question_id));

                // Step 1: Retrieve context
                let ctx = match retrieval::process_question(question, budget) {
                    Ok(ctx) => ctx,
                    Err(e) => {
                        pb.println(format!(
                            "  WARN: retrieval failed for {}: {e}",
                            question.question_id
                        ));
                        error_count += 1;
                        pb.inc(1);
                        continue;
                    }
                };

                // Step 2: Generate answer via Claude CLI (sonnet)
                let prompt = retrieval::build_generation_prompt(
                    &ctx.context_text,
                    &question.question,
                    &question.question_date,
                    question.question_type,
                    question.is_abstention(),
                );

                let hypothesis = match gen_client.complete(&prompt, 512) {
                    Ok(h) => h,
                    Err(e) => {
                        pb.println(format!(
                            "  WARN: generation failed for {}: {e}",
                            question.question_id
                        ));
                        error_count += 1;
                        pb.inc(1);
                        continue;
                    }
                };

                // Step 2b: Self-verification (for counting/arithmetic question types)
                let hypothesis = if no_verify {
                    hypothesis
                } else {
                    match verify::maybe_verify(
                        &gen_client,
                        &ctx.context_text,
                        &question.question,
                        &hypothesis,
                        question.question_type,
                        question.is_abstention(),
                    ) {
                        Ok(h) => h,
                        Err(e) => {
                            pb.println(format!(
                                "  WARN: verification failed for {}, using unverified: {e}",
                                question.question_id
                            ));
                            hypothesis
                        }
                    }
                };

                // Step 3: Judge the answer
                let ground_truth = question.answer.as_text();
                let is_correct = match judge::judge_answer(
                    &judge_client,
                    &question.question,
                    &ground_truth,
                    &hypothesis,
                    question.question_type,
                    question.is_abstention(),
                ) {
                    Ok(c) => c,
                    Err(e) => {
                        pb.println(format!(
                            "  WARN: judging failed for {}: {e}",
                            question.question_id
                        ));
                        error_count += 1;
                        pb.inc(1);
                        continue;
                    }
                };

                if is_correct {
                    correct_count += 1;
                }

                let result = EvalResult {
                    question_id: question.question_id.clone(),
                    question_type: question.question_type,
                    is_abstention: question.is_abstention(),
                    hypothesis,
                    ground_truth,
                    is_correct,
                    tokens_used: 0,
                };

                // Append to results file (JSONL) — incremental save
                let line = serde_json::to_string(&result)?;
                use std::io::Write;
                let mut file = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&output)?;
                writeln!(file, "{line}")?;

                pb.inc(1);

                // Print running accuracy every 25 questions
                let done = total_done + i + 1;
                if done % 25 == 0 {
                    let running = correct_count as f64 / (i + 1) as f64 * 100.0;
                    pb.println(format!(
                        "  [{done}/{}] Running accuracy: {running:.1}% ({error_count} errors)",
                        questions.len()
                    ));
                }
            }

            pb.finish_with_message("done");

            if error_count > 0 {
                println!("\n{error_count} questions had errors and were skipped.");
            }

            // Load ALL results (including previously completed) for final report
            let data = std::fs::read_to_string(&output)?;
            let all_results: Vec<EvalResult> = data
                .lines()
                .filter(|l| !l.trim().is_empty())
                .map(|l| serde_json::from_str(l))
                .collect::<std::result::Result<Vec<_>, _>>()?;

            let m = metrics::compute_metrics(&all_results);
            metrics::print_report(&m);
            println!("\nResults saved to: {output}");
        }
        Commands::Report { results_file } => {
            let data = std::fs::read_to_string(&results_file)?;
            let results: Vec<EvalResult> = data
                .lines()
                .filter(|l| !l.trim().is_empty())
                .map(|l| serde_json::from_str(l))
                .collect::<std::result::Result<Vec<_>, _>>()?;

            let m = metrics::compute_metrics(&results);
            metrics::print_report(&m);
        }
    }

    Ok(())
}

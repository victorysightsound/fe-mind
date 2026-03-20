mod dataset;
mod download;
mod ingest;
mod judge;
mod llm;
mod metrics;
mod retrieval;

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
        /// Output file for results
        #[arg(short, long, default_value = "results.jsonl")]
        output: String,
        /// Context budget in tokens
        #[arg(short, long, default_value = "4096")]
        budget: usize,
        /// Claude model to use (default: subscription default)
        #[arg(short, long)]
        model: Option<String>,
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
        } => {
            let v = parse_variant(&variant);
            let path = download::download_dataset(v).await?;
            let ds = dataset::Dataset::load(&path)?;

            let client = match model {
                Some(m) => llm::ClaudeCliClient::with_model(m),
                None => llm::ClaudeCliClient::new(),
            };

            let questions = if limit > 0 {
                &ds.questions[..limit.min(ds.questions.len())]
            } else {
                &ds.questions
            };

            println!(
                "Running LongMemEval benchmark: {} questions, budget={budget} tokens",
                questions.len()
            );
            println!("Using Claude Code CLI (subscription, no API tokens)\n");

            let pb = ProgressBar::new(questions.len() as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("[{bar:40}] {pos}/{len} ({eta}) {msg}")
                    .expect("style")
                    .progress_chars("=> "),
            );

            let mut results: Vec<EvalResult> = Vec::new();

            // Ensure output directory exists
            if let Some(parent) = Path::new(&output).parent() {
                if !parent.as_os_str().is_empty() {
                    std::fs::create_dir_all(parent)?;
                }
            }

            let mut correct_count = 0_usize;

            for (i, question) in questions.iter().enumerate() {
                pb.set_message(format!("{}", question.question_id));

                // Step 1: Retrieve context from MindCore
                let ctx = retrieval::process_question(question, budget)?;

                // Step 2: Generate answer via Claude CLI
                let prompt = retrieval::build_generation_prompt(
                    &ctx.context_text,
                    &question.question,
                    &question.question_date,
                );

                let hypothesis = client.complete(&prompt, 512)?;

                // Step 3: Judge the answer
                let ground_truth = question.answer.as_text();
                let is_correct = judge::judge_answer(
                    &client,
                    &question.question,
                    &ground_truth,
                    &hypothesis,
                    question.question_type,
                    question.is_abstention(),
                )?;

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
                    tokens_used: 0, // CLI doesn't report token counts
                };

                // Append to results file (JSONL)
                let line = serde_json::to_string(&result)?;
                use std::io::Write;
                let mut file = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&output)?;
                writeln!(file, "{line}")?;

                results.push(result);
                pb.inc(1);

                // Print running accuracy every 10 questions
                if (i + 1) % 10 == 0 {
                    let running = correct_count as f64 / (i + 1) as f64 * 100.0;
                    pb.println(format!("  [{}/{}] Running accuracy: {running:.1}%", i + 1, questions.len()));
                }
            }

            pb.finish_with_message("done");

            // Compute and print metrics
            let m = metrics::compute_metrics(&results);
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

mod dataset;
mod download;

use anyhow::Result;
use clap::{Parser, Subcommand};
use dataset::DatasetVariant;

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
        #[arg(short, long, default_value = "results/results.jsonl")]
        output: String,
    },
    /// Show metrics from a results file
    Report {
        /// Path to results JSONL file
        #[arg(default_value = "results/results.jsonl")]
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
            let total_sessions: usize = ds.questions.iter().map(|q| q.haystack_sessions.len()).sum();
            println!();
            println!("Total sessions: {total_sessions}");
            println!("Total turns: {total_turns}");
        }
        Commands::Run { variant, limit, output } => {
            println!("Benchmark run not yet implemented.");
            println!("Variant: {variant}, Limit: {limit}, Output: {output}");
        }
        Commands::Report { results_file } => {
            println!("Report not yet implemented.");
            println!("Results file: {results_file}");
        }
    }

    Ok(())
}

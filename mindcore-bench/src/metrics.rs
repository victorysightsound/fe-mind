use std::collections::HashMap;

use crate::dataset::{EvalResult, QuestionType};

/// Computed benchmark metrics.
pub struct BenchmarkMetrics {
    /// Accuracy per question type (6 types).
    pub per_type: HashMap<QuestionType, TypeMetric>,
    /// Task-averaged accuracy (mean of per-type accuracies).
    pub task_averaged: f64,
    /// Overall accuracy (mean of all questions).
    pub overall: f64,
    /// Abstention accuracy (30 questions).
    pub abstention: f64,
    /// Total questions evaluated.
    pub total_questions: usize,
    /// Total tokens used.
    pub total_tokens: u64,
}

/// Per-type accuracy metric.
pub struct TypeMetric {
    pub correct: usize,
    pub total: usize,
    pub accuracy: f64,
}

/// Compute metrics from evaluation results.
pub fn compute_metrics(results: &[EvalResult]) -> BenchmarkMetrics {
    let mut per_type: HashMap<QuestionType, (usize, usize)> = HashMap::new();
    let mut abs_correct = 0_usize;
    let mut abs_total = 0_usize;
    let mut total_correct = 0_usize;
    let mut total_tokens = 0_u64;

    for r in results {
        total_tokens += r.tokens_used as u64;

        if r.is_abstention {
            abs_total += 1;
            if r.is_correct {
                abs_correct += 1;
            }
        }

        let entry = per_type.entry(r.question_type).or_insert((0, 0));
        entry.1 += 1;
        if r.is_correct {
            entry.0 += 1;
            total_correct += 1;
        }
    }

    let type_metrics: HashMap<QuestionType, TypeMetric> = per_type
        .into_iter()
        .map(|(qt, (correct, total))| {
            let accuracy = if total > 0 {
                correct as f64 / total as f64
            } else {
                0.0
            };
            (
                qt,
                TypeMetric {
                    correct,
                    total,
                    accuracy,
                },
            )
        })
        .collect();

    let task_averaged = if type_metrics.is_empty() {
        0.0
    } else {
        type_metrics.values().map(|m| m.accuracy).sum::<f64>() / type_metrics.len() as f64
    };

    let overall = if results.is_empty() {
        0.0
    } else {
        total_correct as f64 / results.len() as f64
    };

    let abstention = if abs_total > 0 {
        abs_correct as f64 / abs_total as f64
    } else {
        0.0
    };

    BenchmarkMetrics {
        per_type: type_metrics,
        task_averaged,
        overall,
        abstention,
        total_questions: results.len(),
        total_tokens,
    }
}

/// Print a formatted report.
pub fn print_report(metrics: &BenchmarkMetrics) {
    println!("\nLongMemEval Benchmark Results");
    println!("============================\n");

    println!("Per-Type Accuracy:");
    let mut types: Vec<_> = metrics.per_type.iter().collect();
    types.sort_by_key(|(qt, _)| format!("{qt:?}"));
    for (qt, m) in &types {
        println!(
            "  {:<30} {:>3}/{:<3} ({:.1}%)",
            qt.display_name(),
            m.correct,
            m.total,
            m.accuracy * 100.0
        );
    }

    println!();
    println!(
        "Task-Averaged Accuracy:  {:.1}%",
        metrics.task_averaged * 100.0
    );
    println!("Overall Accuracy:        {:.1}%", metrics.overall * 100.0);
    println!(
        "Abstention Accuracy:     {:.1}%",
        metrics.abstention * 100.0
    );
    println!();
    println!("Total Questions: {}", metrics.total_questions);
    println!("Total Tokens:    {}", metrics.total_tokens);
}

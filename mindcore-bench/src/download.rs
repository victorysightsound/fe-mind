use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};

use crate::dataset::DatasetVariant;

/// Default data directory for downloaded datasets.
pub fn data_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home)
        .join(".cache")
        .join("mindcore-bench")
        .join("data")
}

/// Download a dataset variant if not already cached.
///
/// Returns the path to the local file.
pub async fn download_dataset(variant: DatasetVariant) -> Result<PathBuf> {
    let dir = data_dir();
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("failed to create data dir: {}", dir.display()))?;

    let local_path = dir.join(variant.filename());

    if local_path.exists() {
        let size = std::fs::metadata(&local_path)?.len();
        if size > 1000 {
            tracing::info!(
                "Dataset already cached: {} ({:.1}MB)",
                local_path.display(),
                size as f64 / 1_000_000.0
            );
            return Ok(local_path);
        }
    }

    let url = variant.download_url();
    tracing::info!("Downloading {} from {}", variant.filename(), url);

    let client = reqwest::Client::new();
    let response = client
        .get(&url)
        .send()
        .await
        .with_context(|| format!("failed to request {url}"))?;

    let total_size = response.content_length().unwrap_or(0);

    let pb = ProgressBar::new(total_size);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
            .expect("progress style")
            .progress_chars("#>-"),
    );

    let bytes = response
        .bytes()
        .await
        .with_context(|| "failed to download dataset")?;

    pb.finish_with_message("Download complete");

    std::fs::write(&local_path, &bytes)
        .with_context(|| format!("failed to write {}", local_path.display()))?;

    tracing::info!(
        "Saved {} ({:.1}MB)",
        local_path.display(),
        bytes.len() as f64 / 1_000_000.0
    );

    Ok(local_path)
}

/// Check if a dataset is already cached locally.
pub fn is_cached(variant: DatasetVariant) -> bool {
    let path = data_dir().join(variant.filename());
    path.exists() && std::fs::metadata(&path).map(|m| m.len() > 1000).unwrap_or(false)
}

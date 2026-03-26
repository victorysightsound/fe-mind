//! CliLlmCallback: wraps Claude/ChatGPT/Gemini CLI tools.
//!
//! Pipes prompt via stdin, reads response from stdout.
//! Works with any CLI tool that accepts text input and produces text output.

use std::path::PathBuf;

use crate::error::{FemindError, Result};
use crate::traits::LlmCallback;

/// LLM callback using a CLI tool (claude, chatgpt, gemini, etc.)
pub struct CliLlmCallback {
    command: String,
    args: Vec<String>,
    model_label: String,
}

impl CliLlmCallback {
    /// Create a CLI callback for Claude.
    pub fn claude(model: &str) -> Self {
        let model_arg = match model {
            "sonnet" | "claude-sonnet" => "sonnet",
            "opus" | "claude-opus" => "opus",
            "haiku" | "claude-haiku" => "haiku",
            other => other,
        };
        let mut args = vec!["--print".to_string()];
        if model_arg != "sonnet" {
            args.extend(["--model".to_string(), model_arg.to_string()]);
        }
        Self {
            command: "claude".to_string(),
            args,
            model_label: format!("claude-{model_arg}"),
        }
    }

    /// Create a CLI callback for Codex exec.
    pub fn codex(model: &str) -> Self {
        Self {
            command: "codex".to_string(),
            args: vec![
                "exec".to_string(),
                "--model".to_string(),
                model.to_string(),
                "--skip-git-repo-check".to_string(),
                "--sandbox".to_string(),
                "read-only".to_string(),
                "--ephemeral".to_string(),
                "-".to_string(),
            ],
            model_label: format!("codex-{model}"),
        }
    }

    /// Create a CLI callback for any command.
    pub fn custom(
        command: impl Into<String>,
        args: Vec<String>,
        model_label: impl Into<String>,
    ) -> Self {
        Self {
            command: command.into(),
            args,
            model_label: model_label.into(),
        }
    }
}

impl LlmCallback for CliLlmCallback {
    fn generate(&self, prompt: &str, _max_tokens: usize) -> Result<String> {
        let mut cmd = std::process::Command::new(&self.command);
        cmd.env("RECALLBENCH_SUBPROCESS", "1"); // Signal to hooks
        let mut args = self.args.clone();
        let codex_output_path = if self.command == "codex" {
            let path = codex_output_path();
            args.splice(
                1..1,
                [
                    "--output-last-message".to_string(),
                    path.to_string_lossy().into_owned(),
                ],
            );
            Some(path)
        } else {
            None
        };
        cmd.args(&args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());

        let mut child = cmd.spawn().map_err(|e| {
            FemindError::Embedding(format!("Failed to spawn '{}': {e}", self.command))
        })?;

        if let Some(stdin) = child.stdin.as_mut() {
            use std::io::Write;
            stdin
                .write_all(prompt.as_bytes())
                .map_err(|e| FemindError::Embedding(format!("stdin write: {e}")))?;
        }

        let output = child
            .wait_with_output()
            .map_err(|e| FemindError::Embedding(format!("wait: {e}")))?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let codex_output = codex_output_path
            .as_ref()
            .and_then(|path| std::fs::read_to_string(path).ok())
            .map(|text| text.trim().to_string())
            .filter(|text| !text.is_empty());
        if let Some(path) = codex_output_path {
            let _ = std::fs::remove_file(path);
        }
        let text_output = codex_output.unwrap_or_else(|| stdout.trim().to_string());

        // Tolerate non-zero exit if stdout has content (hooks may fail)
        if text_output.trim().is_empty() && !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(FemindError::Embedding(format!(
                "{} exited with {}: {}",
                self.command,
                output.status,
                stderr.trim()
            )));
        }

        Ok(text_output)
    }

    fn model_name(&self) -> &str {
        &self.model_label
    }
}

fn codex_output_path() -> PathBuf {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!(
        "femind-codex-output-{}-{stamp}.txt",
        std::process::id()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn claude_args_default() {
        let cli = CliLlmCallback::claude("sonnet");
        assert_eq!(cli.command, "claude");
        assert!(cli.args.contains(&"--print".to_string()));
        assert_eq!(cli.model_label, "claude-sonnet");
    }

    #[test]
    fn claude_args_haiku() {
        let cli = CliLlmCallback::claude("haiku");
        assert!(cli.args.contains(&"haiku".to_string()));
        assert_eq!(cli.model_label, "claude-haiku");
    }

    #[test]
    fn codex_args_include_exec_and_model() {
        let cli = CliLlmCallback::codex("gpt-5.4-mini");
        assert_eq!(cli.command, "codex");
        assert!(cli.args.contains(&"exec".to_string()));
        assert!(cli.args.contains(&"--model".to_string()));
        assert!(cli.args.contains(&"gpt-5.4-mini".to_string()));
        assert_eq!(cli.model_label, "codex-gpt-5.4-mini");
    }
}

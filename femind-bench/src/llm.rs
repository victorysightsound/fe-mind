use anyhow::{Context, Result};
use std::process::Command;

/// LLM client that shells out to the Claude Code CLI.
///
/// Uses `claude --print` for non-interactive single-prompt completion.
/// No API key needed — uses the existing Claude Code subscription.
pub struct ClaudeCliClient {
    /// Model to use (passed via --model flag, or omit for default).
    model: Option<String>,
}

impl ClaudeCliClient {
    /// Create with the default model.
    pub fn new() -> Self {
        Self { model: None }
    }

    /// Create with a specific model.
    pub fn with_model(model: impl Into<String>) -> Self {
        Self {
            model: Some(model.into()),
        }
    }

    /// Send a prompt and get a response via `claude --print`.
    ///
    /// Returns the response text.
    pub fn complete(&self, prompt: &str, _max_tokens: u32) -> Result<String> {
        let mut cmd = Command::new("claude");
        cmd.arg("--print");

        if let Some(ref model) = self.model {
            cmd.arg("--model").arg(model);
        }

        // Pass prompt via stdin to avoid shell escaping issues
        cmd.stdin(std::process::Stdio::piped());
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());

        let mut child = cmd
            .spawn()
            .context("failed to spawn claude CLI — is it installed?")?;

        // Write prompt to stdin
        if let Some(ref mut stdin) = child.stdin {
            use std::io::Write;
            stdin
                .write_all(prompt.as_bytes())
                .context("failed to write to claude stdin")?;
        }
        // Close stdin so claude knows input is complete
        drop(child.stdin.take());

        let output = child
            .wait_with_output()
            .context("failed to wait for claude CLI")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("claude CLI failed ({}): {stderr}", output.status);
        }

        let response = String::from_utf8(output.stdout)
            .context("claude CLI output is not valid UTF-8")?
            .trim()
            .to_string();

        Ok(response)
    }
}

impl Default for ClaudeCliClient {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn client_creates() {
        let client = ClaudeCliClient::new();
        assert!(client.model.is_none());
    }

    #[test]
    fn client_with_model() {
        let client = ClaudeCliClient::with_model("sonnet");
        assert_eq!(client.model.as_deref(), Some("sonnet"));
    }
}

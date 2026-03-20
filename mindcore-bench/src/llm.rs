use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Simple Anthropic API client for Claude.
pub struct ClaudeClient {
    api_key: String,
    client: reqwest::Client,
    model: String,
    pub total_input_tokens: std::sync::atomic::AtomicU32,
    pub total_output_tokens: std::sync::atomic::AtomicU32,
}

#[derive(Serialize)]
struct MessageRequest {
    model: String,
    max_tokens: u32,
    temperature: f32,
    messages: Vec<Message>,
}

#[derive(Serialize, Deserialize, Clone)]
struct Message {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct MessageResponse {
    content: Vec<ContentBlock>,
    usage: Usage,
}

#[derive(Deserialize)]
struct ContentBlock {
    text: String,
}

#[derive(Deserialize)]
struct Usage {
    input_tokens: u32,
    output_tokens: u32,
}

impl ClaudeClient {
    /// Create from API key. Uses Claude Sonnet by default.
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            client: reqwest::Client::new(),
            model: "claude-sonnet-4-20250514".into(),
            total_input_tokens: std::sync::atomic::AtomicU32::new(0),
            total_output_tokens: std::sync::atomic::AtomicU32::new(0),
        }
    }

    /// Create from environment variable.
    pub fn from_env() -> Result<Self> {
        let key = std::env::var("ANTHROPIC_API_KEY")
            .context("ANTHROPIC_API_KEY environment variable not set")?;
        Ok(Self::new(key))
    }

    /// Send a message and get a response.
    pub async fn complete(&self, prompt: &str, max_tokens: u32) -> Result<(String, u32)> {
        let request = MessageRequest {
            model: self.model.clone(),
            max_tokens,
            temperature: 0.0,
            messages: vec![Message {
                role: "user".into(),
                content: prompt.into(),
            }],
        };

        let response = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&request)
            .send()
            .await
            .context("failed to send request to Anthropic API")?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("Anthropic API error {status}: {body}");
        }

        let msg: MessageResponse = response
            .json()
            .await
            .context("failed to parse Anthropic response")?;

        let text = msg
            .content
            .first()
            .map(|b| b.text.clone())
            .unwrap_or_default();

        let tokens = msg.usage.input_tokens + msg.usage.output_tokens;
        self.total_input_tokens
            .fetch_add(msg.usage.input_tokens, std::sync::atomic::Ordering::Relaxed);
        self.total_output_tokens
            .fetch_add(msg.usage.output_tokens, std::sync::atomic::Ordering::Relaxed);

        Ok((text, tokens))
    }

    /// Total tokens used across all calls.
    pub fn total_tokens(&self) -> u32 {
        self.total_input_tokens.load(std::sync::atomic::Ordering::Relaxed)
            + self.total_output_tokens.load(std::sync::atomic::Ordering::Relaxed)
    }
}

use std::collections::HashMap;
use std::env;
use std::process::ExitCode;

use chrono::{DateTime, Utc};
use femind::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ReviewMemory {
    id: Option<i64>,
    text: String,
    created_at: DateTime<Utc>,
    metadata: HashMap<String, String>,
}

impl MemoryRecord for ReviewMemory {
    fn id(&self) -> Option<i64> {
        self.id
    }

    fn searchable_text(&self) -> String {
        self.text.clone()
    }

    fn memory_type(&self) -> MemoryType {
        MemoryType::Procedural
    }

    fn created_at(&self) -> DateTime<Utc> {
        self.created_at
    }

    fn metadata(&self) -> HashMap<String, String> {
        self.metadata.clone()
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("femind-review: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> femind::error::Result<()> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    let Some(command) = args.first().cloned() else {
        print_usage();
        return Ok(());
    };
    let command_args = &args[1..];

    match command.as_str() {
        "list" => cmd_list(command_args),
        "resolve" => cmd_resolve(command_args),
        "renew" => cmd_renew(command_args),
        "revoke" => cmd_revoke(command_args),
        "replace" => cmd_replace(command_args),
        "expire-due" => cmd_expire_due(command_args),
        "--help" | "-h" | "help" => {
            print_usage();
            Ok(())
        }
        other => Err(FemindError::Migration(format!(
            "unknown command '{other}'. Use 'femind-review help' for usage."
        ))),
    }
}

fn cmd_list(args: &[String]) -> femind::error::Result<()> {
    let database = required_arg(args, "--database")?;
    let limit = optional_arg(args, "--limit")
        .map(|value| parse_usize("--limit", &value))
        .transpose()?
        .unwrap_or(25);
    let format = optional_arg(args, "--format").unwrap_or_else(|| "text".to_string());
    let status = optional_arg(args, "--status")
        .map(|value| parse_status_filter(&value))
        .transpose()?;

    let engine = open_engine(&database)?;
    let items = match status {
        Some(filter) => engine.review_items_with_status(limit, Some(filter))?,
        None => engine.review_items(limit)?,
    };

    if format.eq_ignore_ascii_case("json") {
        println!("{}", serde_json::to_string_pretty(&items)?);
    } else {
        for item in items {
            let note = item
                .note
                .as_deref()
                .map(|value| format!(" note={value:?}"))
                .unwrap_or_default();
            let reviewer = item
                .reviewer
                .as_deref()
                .map(|value| format!(" reviewer={value}"))
                .unwrap_or_default();
            let scope = item
                .scope
                .map(|value| format!(" scope={value}"))
                .unwrap_or_default();
            let policy_class = item
                .policy_class
                .map(|value| format!(" class={value}"))
                .unwrap_or_default();
            let template = item
                .template
                .map(|value| format!(" template={value}"))
                .unwrap_or_default();
            let replaced_by = item
                .replaced_by
                .map(|value| format!(" replaced_by=#{value}"))
                .unwrap_or_default();
            let expires = item
                .expires_at
                .map(|value| format!(" expires_at={}", value.to_rfc3339()))
                .unwrap_or_default();
            println!(
                "#{id} [{status}] severity={severity} created_at={created_at} updated_at={updated_at}{expires}{scope}{policy_class}{template}{reviewer}{replaced_by}{note}\n  reason={reason}\n  tags={tags}\n  text={text}",
                id = item.memory_id,
                status = item.status,
                severity = item.severity,
                created_at = item.created_at.to_rfc3339(),
                updated_at = item
                    .updated_at
                    .map(|value| value.to_rfc3339())
                    .unwrap_or_else(|| "-".to_string()),
                reason = item.reason,
                tags = if item.tags.is_empty() {
                    "-".to_string()
                } else {
                    item.tags.join(",")
                },
                text = item.text
            );
        }
    }

    Ok(())
}

fn cmd_resolve(args: &[String]) -> femind::error::Result<()> {
    let database = required_arg(args, "--database")?;
    let memory_id = required_arg(args, "--memory-id")?
        .parse::<i64>()
        .map_err(|_| FemindError::Migration("invalid --memory-id".to_string()))?;
    let status = parse_status_filter(&required_arg(args, "--status")?)?;
    let note = optional_arg(args, "--note");
    let reviewer = optional_arg(args, "--reviewer");
    let scope = optional_arg(args, "--scope")
        .map(|value| parse_scope("--scope", &value))
        .transpose()?;
    let template = optional_arg(args, "--template")
        .map(|value| parse_template("--template", &value))
        .transpose()?;
    let policy_class = optional_arg(args, "--class")
        .map(|value| parse_policy_class("--class", &value))
        .transpose()?;
    let replacement_id = optional_arg(args, "--replacement-id")
        .map(|value| parse_i64("--replacement-id", &value))
        .transpose()?;
    let expires_at = optional_arg(args, "--expires-at")
        .map(|value| parse_datetime("--expires-at", &value))
        .transpose()?;
    let format = optional_arg(args, "--format").unwrap_or_else(|| "text".to_string());

    let engine = open_engine(&database)?;
    let item = engine.resolve_review_item_with_resolution(
        memory_id,
        femind::engine::ReviewResolution {
            status,
            note,
            reviewer,
            scope,
            policy_class,
            template,
            expires_at,
            replaced_by: replacement_id,
        },
    )?;

    print_review_resolution(&item, &format, "resolved")
}

fn cmd_renew(args: &[String]) -> femind::error::Result<()> {
    let database = required_arg(args, "--database")?;
    let memory_id = parse_i64("--memory-id", &required_arg(args, "--memory-id")?)?;
    let note = optional_arg(args, "--note");
    let reviewer = optional_arg(args, "--reviewer");
    let expires_at = optional_arg(args, "--expires-at")
        .map(|value| parse_datetime("--expires-at", &value))
        .transpose()?;
    let format = optional_arg(args, "--format").unwrap_or_else(|| "text".to_string());

    let engine = open_engine(&database)?;
    let item =
        engine.renew_review_item(memory_id, reviewer.as_deref(), note.as_deref(), expires_at)?;

    print_review_resolution(&item, &format, "renewed")
}

fn cmd_revoke(args: &[String]) -> femind::error::Result<()> {
    let database = required_arg(args, "--database")?;
    let memory_id = parse_i64("--memory-id", &required_arg(args, "--memory-id")?)?;
    let note = optional_arg(args, "--note");
    let reviewer = optional_arg(args, "--reviewer");
    let format = optional_arg(args, "--format").unwrap_or_else(|| "text".to_string());

    let engine = open_engine(&database)?;
    let item = engine.revoke_review_item(memory_id, reviewer.as_deref(), note.as_deref())?;

    print_review_resolution(&item, &format, "revoked")
}

fn cmd_replace(args: &[String]) -> femind::error::Result<()> {
    let database = required_arg(args, "--database")?;
    let memory_id = parse_i64("--memory-id", &required_arg(args, "--memory-id")?)?;
    let replacement_id = parse_i64("--replacement-id", &required_arg(args, "--replacement-id")?)?;
    let note = optional_arg(args, "--note");
    let reviewer = optional_arg(args, "--reviewer");
    let format = optional_arg(args, "--format").unwrap_or_else(|| "text".to_string());

    let engine = open_engine(&database)?;
    let item = engine.replace_review_item(
        memory_id,
        replacement_id,
        reviewer.as_deref(),
        note.as_deref(),
    )?;

    print_review_resolution(&item, &format, "replaced")
}

fn cmd_expire_due(args: &[String]) -> femind::error::Result<()> {
    let database = required_arg(args, "--database")?;
    let now = optional_arg(args, "--now")
        .map(|value| parse_datetime("--now", &value))
        .transpose()?
        .unwrap_or_else(Utc::now);
    let format = optional_arg(args, "--format").unwrap_or_else(|| "text".to_string());

    let engine = open_engine(&database)?;
    let expired = engine.expire_due_review_items(now)?;

    if format.eq_ignore_ascii_case("json") {
        println!(
            "{}",
            serde_json::json!({
                "expired": expired,
                "at": now.to_rfc3339(),
            })
        );
    } else {
        println!("expired {expired} review item(s) at {}", now.to_rfc3339());
    }

    Ok(())
}

fn open_engine(database: &str) -> femind::error::Result<MemoryEngine<ReviewMemory>> {
    MemoryEngine::<ReviewMemory>::builder()
        .database(database.to_string())
        .build()
}

fn required_arg(args: &[String], flag: &str) -> femind::error::Result<String> {
    optional_arg(args, flag).ok_or_else(|| {
        FemindError::Migration(format!(
            "missing required {flag}. Use 'femind-review help' for usage."
        ))
    })
}

fn optional_arg(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|value| value == flag)
        .and_then(|index| args.get(index + 1))
        .cloned()
}

fn parse_status_filter(value: &str) -> femind::error::Result<ReviewStatus> {
    ReviewStatus::from_str(value).ok_or_else(|| {
        FemindError::Migration(format!(
            "invalid review status '{value}'. Expected pending|allowed|denied|expired."
        ))
    })
}

fn parse_scope(flag: &str, value: &str) -> femind::error::Result<ReviewScope> {
    ReviewScope::from_str(value).ok_or_else(|| {
        FemindError::Migration(format!(
            "invalid {flag} value '{value}'. Expected general|production|staging|lab|migration."
        ))
    })
}

fn parse_policy_class(flag: &str, value: &str) -> femind::error::Result<ReviewPolicyClass> {
    ReviewPolicyClass::from_str(value).ok_or_else(|| {
        FemindError::Migration(format!(
            "invalid {flag} value '{value}'. Expected operational-exception|network-exposure-exception|destructive-maintenance|secret-handling-exception|migration-exception."
        ))
    })
}

fn parse_template(flag: &str, value: &str) -> femind::error::Result<ReviewApprovalTemplate> {
    ReviewApprovalTemplate::from_str(value).ok_or_else(|| {
        FemindError::Migration(format!(
            "invalid {flag} value '{value}'. Expected staging-bridge|migration-bridge|lab-exception."
        ))
    })
}

fn parse_datetime(flag: &str, value: &str) -> femind::error::Result<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(value)
        .map(|value| value.with_timezone(&Utc))
        .map_err(|_| FemindError::Migration(format!("invalid {flag} datetime '{value}'")))
}

fn parse_i64(flag: &str, value: &str) -> femind::error::Result<i64> {
    value
        .parse::<i64>()
        .map_err(|_| FemindError::Migration(format!("invalid {flag} value '{value}'")))
}

fn parse_usize(flag: &str, value: &str) -> femind::error::Result<usize> {
    value
        .parse::<usize>()
        .map_err(|_| FemindError::Migration(format!("invalid {flag} value '{value}'")))
}

fn print_usage() {
    println!(
        "\
femind-review

Usage:
  femind-review list --database <path> [--status <pending|allowed|denied|expired>] [--limit <n>] [--format <text|json>]
  femind-review resolve --database <path> --memory-id <id> --status <pending|allowed|denied|expired> [--note <text>] [--reviewer <name>] [--scope <general|production|staging|lab|migration>] [--class <policy-class>] [--template <staging-bridge|migration-bridge|lab-exception>] [--replacement-id <id>] [--expires-at <rfc3339>] [--format <text|json>]
  femind-review renew --database <path> --memory-id <id> [--note <text>] [--reviewer <name>] [--expires-at <rfc3339>] [--format <text|json>]
  femind-review revoke --database <path> --memory-id <id> [--note <text>] [--reviewer <name>] [--format <text|json>]
  femind-review replace --database <path> --memory-id <id> --replacement-id <id> [--note <text>] [--reviewer <name>] [--format <text|json>]
  femind-review expire-due --database <path> [--now <rfc3339>] [--format <text|json>]
"
    );
}

fn print_review_resolution(
    item: &femind::engine::ReviewItem,
    format: &str,
    action: &str,
) -> femind::error::Result<()> {
    if format.eq_ignore_ascii_case("json") {
        println!("{}", serde_json::to_string_pretty(item)?);
    } else {
        println!(
            "{action} #{id} as {status} (expires_at={expires})",
            id = item.memory_id,
            status = item.status,
            expires = item
                .expires_at
                .map(|value| value.to_rfc3339())
                .unwrap_or_else(|| "-".to_string())
        );
    }

    Ok(())
}

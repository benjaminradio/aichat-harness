mod markdown;
mod stream;

pub use self::markdown::{MarkdownRender, RenderOptions};
use self::stream::{markdown_stream, raw_stream};

use crate::utils::{error_text, pretty_error, AbortSignal, IS_STDOUT_TERMINAL};
use crate::{client::SseEvent, config::GlobalConfig};

use anyhow::Result;
use tokio::sync::mpsc::UnboundedReceiver;

/// Wrap reasoning text for display. Replaces the old practice of baking `<think>`
/// tags directly into the streamed/stored text — reasoning now lives in
/// `Message.reasoning_content`, kept out of the text a provider might re-parse or
/// re-send, and gets this formatting applied only at print time (here, and in
/// `render/stream.rs` for live streaming).
pub fn format_reasoning(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }
    format!("<think>\n{text}\n</think>\n")
}

/// Render a stored `spawn_subagent` trace for `.info session`/session replay. `trace`
/// is the `{"agent": ..., "messages": [...]}` value `eval_spawn_subagent` records;
/// `fallback` is the tool result's own terse text, used if `trace` is missing or
/// doesn't parse as expected. Deliberately a flat summary (agent name + each
/// non-system message's text, joined) rather than a fully recursive nested render —
/// enough to inspect what a subagent did without reimplementing this function's own
/// caller inside itself.
pub fn format_subagent_trace(trace: Option<&serde_json::Value>, fallback: &str) -> String {
    let trace = match trace {
        Some(v) => v,
        None => return fallback.to_string(),
    };
    let agent = trace.get("agent").and_then(|v| v.as_str()).unwrap_or("?");
    let body = match trace.get("messages").and_then(|v| v.as_array()) {
        Some(messages) => messages
            .iter()
            .filter(|m| m.get("role").and_then(|v| v.as_str()) != Some("system"))
            .filter_map(|m| m.get("content").and_then(|v| v.as_str()))
            .filter(|v| !v.is_empty())
            .collect::<Vec<_>>()
            .join("\n\n"),
        None => fallback.to_string(),
    };
    format!("<subagent agent=\"{agent}\">\n{body}\n</subagent>")
}

pub async fn render_stream(
    rx: UnboundedReceiver<SseEvent>,
    config: &GlobalConfig,
    abort_signal: AbortSignal,
) -> Result<()> {
    let ret = if *IS_STDOUT_TERMINAL && config.read().highlight {
        let render_options = config.read().render_options()?;
        let mut render = MarkdownRender::init(render_options)?;
        markdown_stream(rx, &mut render, &abort_signal).await
    } else {
        raw_stream(rx, &abort_signal).await
    };
    ret.map_err(|err| err.context("Failed to reader stream"))
}

pub fn render_error(err: anyhow::Error) {
    eprintln!("{}", error_text(&pretty_error(&err)));
}

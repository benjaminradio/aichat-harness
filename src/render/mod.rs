mod markdown;
mod stream;

pub use self::markdown::{MarkdownRender, RenderOptions};
use self::stream::{drain_stream, markdown_stream, raw_stream};

use crate::utils::{dimmed_text, error_text, pretty_error, AbortSignal, IS_STDOUT_TERMINAL};
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
///
/// The `<subagent ...>`/`</subagent>` boundary lines are dimmed (`dimmed_text`),
/// matching `eval_spawn_subagent`'s own live-print treatment exactly — this is the
/// *other* place that same boundary text gets produced, so it needs the same
/// styling, not just the live one, or replaying a session still shows it as flat,
/// unstyled text.
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
    let open = dimmed_text(&format!("<subagent agent=\"{agent}\">"));
    let close = dimmed_text("</subagent>");
    format!("{open}\n{body}\n{close}")
}

/// Displays a completion's live stream, or -- for a spawned subagent's own
/// stream, when `show_subagent` is off (`agent_depth > 0` is exactly "this
/// config belongs to some depth of subagent", set once in
/// `crate::function::eval_spawn_subagent` and inherited by any of its own
/// descendants) -- silently drains it instead. Either way the caller gets
/// the same correct output/reasoning text back; this only ever decides what
/// appears on the terminal.
pub async fn render_stream(
    rx: UnboundedReceiver<SseEvent>,
    config: &GlobalConfig,
    abort_signal: AbortSignal,
) -> Result<()> {
    let suppressed = {
        let cfg = config.read();
        cfg.agent_depth > 0 && !cfg.show_subagent
    };
    if suppressed {
        return drain_stream(rx, &abort_signal).await;
    }
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

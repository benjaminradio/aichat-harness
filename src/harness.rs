use crate::client::{
    call_chat_completions, call_chat_completions_streaming, Message, MessageContent, MessageRole,
};
use crate::config::{GlobalConfig, Input};
use crate::utils::AbortSignal;

use anyhow::Result;

/// The shared "call model, run any tool calls, loop until done" body.
///
/// Before this existed the same logic lived twice: `main.rs::start_directive` (CLI)
/// and `repl/mod.rs::ask` (REPL). Both are now thin wrappers around this, and §7's
/// `eval_spawn_subagent` becomes a third caller (against an isolated `Config` clone)
/// rather than a third copy.
///
/// Differences between the two original call sites that are parameterized here:
/// - `extract_code`: CLI's code mode passes `true`, which also forces the
///   non-streaming path; REPL always passes `false`.
///
/// Differences deliberately left *outside* this function, in the wrappers, because
/// they are not part of the loop itself:
/// - CLI calls `exit_session()` afterward.
/// - REPL calls `maybe_autoname_session`/`maybe_compress_session` on the final turn
///   only (i.e. when no tool calls came back), and does the `is_compressing_session`
///   wait and `use_embeddings` up front.
///
/// Written as a loop rather than the original recursion so it needs no
/// `#[async_recursion]` and the `max_subagent_turns` cap in §7 has an obvious place
/// to hook in via `max_turns`.
///
/// `max_turns`: `None` = unlimited (today's behavior for both existing callers).
/// `Some(n)` force-stops after `n` model round-trips, returning whatever text the
/// last completion produced. Only §7's subagent path passes `Some`.
pub struct CompletionOutcome {
    pub output: String,
    pub reasoning_content: Option<String>,
    /// Every message produced across every round of the loop — the same shape
    /// `Session::add_message` would extend `session.messages` with, plus one final
    /// plain Assistant message for the last round's own text/reasoning. Top-level
    /// callers (`start_directive`/`ask`) ignore this; it exists for §7's
    /// `eval_spawn_subagent`, which has no session to persist into and needs this to
    /// build the subagent's `trace`.
    pub messages: Vec<Message>,
}

pub async fn run_completion_loop(
    config: &GlobalConfig,
    mut input: Input,
    extract_code: bool,
    max_turns: Option<usize>,
    abort_signal: AbortSignal,
) -> Result<CompletionOutcome> {
    let mut turns = 0;
    loop {
        let client = input.create_client()?;
        config.write().before_chat_completion(&input)?;
        let (output, reasoning_content, tool_results) = if !input.stream() || extract_code {
            call_chat_completions(
                &input,
                true,
                extract_code,
                client.as_ref(),
                abort_signal.clone(),
            )
            .await?
        } else {
            call_chat_completions_streaming(&input, client.as_ref(), abort_signal.clone()).await?
        };
        config
            .write()
            .after_chat_completion(&input, &output, &reasoning_content, &tool_results)?;

        turns += 1;

        let done = tool_results.is_empty();
        let hit_max_turns = max_turns.is_some_and(|max| turns >= max);
        if done || hit_max_turns {
            let mut messages = input.pending_messages().to_vec();
            let mut final_message =
                Message::new(MessageRole::Assistant, MessageContent::Text(output.clone()));
            final_message.reasoning_content = reasoning_content.clone();
            messages.push(final_message);
            return Ok(CompletionOutcome {
                output,
                reasoning_content,
                messages,
            });
        }

        input = input.merge_tool_results(output, reasoning_content, tool_results);
    }
}

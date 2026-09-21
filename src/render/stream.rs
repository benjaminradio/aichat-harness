use super::{MarkdownRender, SseEvent};

use crate::utils::{poll_abort_signal, spawn_spinner, AbortSignal};

use anyhow::Result;
use crossterm::{
    cursor, queue, style,
    terminal::{self, disable_raw_mode, enable_raw_mode},
};
use std::{
    io::{self, stdout, Stdout, Write},
    time::Duration,
};
use textwrap::core::display_width;
use tokio::sync::mpsc::UnboundedReceiver;

pub async fn markdown_stream(
    rx: UnboundedReceiver<SseEvent>,
    render: &mut MarkdownRender,
    abort_signal: &AbortSignal,
) -> Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();

    let ret = markdown_stream_inner(rx, render, abort_signal, &mut stdout).await;

    disable_raw_mode()?;

    if ret.is_err() {
        println!();
    }
    ret
}

pub async fn raw_stream(
    mut rx: UnboundedReceiver<SseEvent>,
    abort_signal: &AbortSignal,
) -> Result<()> {
    let mut spinner = Some(spawn_spinner("Generating"));

    loop {
        if abort_signal.aborted() {
            break;
        }
        if let Some(evt) = rx.recv().await {
            match evt {
                SseEvent::ReasoningStart => {
                    if let Some(spinner) = &spinner {
                        let _ = spinner.set_message("Thinking".to_string());
                    }
                }
                SseEvent::Text(text) => {
                    if let Some(spinner) = spinner.take() {
                        spinner.stop();
                    }
                    print!("{text}");
                    stdout().flush()?;
                }
                SseEvent::Done => {
                    if let Some(spinner) = spinner.take() {
                        spinner.stop();
                    }
                    break;
                }
            }
        }
    }
    if let Some(spinner) = spinner.take() {
        spinner.stop();
    }
    Ok(())
}

async fn markdown_stream_inner(
    mut rx: UnboundedReceiver<SseEvent>,
    render: &mut MarkdownRender,
    abort_signal: &AbortSignal,
    writer: &mut Stdout,
) -> Result<()> {
    let mut buffer = String::new();
    let mut buffer_rows = 1;

    let columns = terminal::size()?.0;

    let mut spinner = Some(spawn_spinner("Generating"));

    'outer: loop {
        if abort_signal.aborted() {
            break;
        }
        for reply_event in gather_events(&mut rx).await {
            match reply_event {
                SseEvent::ReasoningStart => {
                    if let Some(spinner) = &spinner {
                        let _ = spinner.set_message("Thinking".to_string());
                    }
                }
                SseEvent::Text(mut text) => {
                    if let Some(spinner) = spinner.take() {
                        spinner.stop();
                    }

                    // tab width hacking
                    text = text.replace('\t', "    ");

                    let mut attempts = 0;
                    let (col, mut row) = loop {
                        match cursor::position() {
                            Ok(pos) => break pos,
                            Err(_) if attempts < 3 => attempts += 1,
                            Err(e) => return Err(e.into()),
                        }
                    };

                    // Fix unexpected duplicate lines on kitty, see https://github.com/sigoden/aichat/issues/105
                    if col == 0 && row > 0 && display_width(&buffer) == columns as usize {
                        row -= 1;
                    }

                    if row + 1 >= buffer_rows {
                        queue!(writer, cursor::MoveTo(0, row + 1 - buffer_rows),)?;
                    } else {
                        let scroll_rows = buffer_rows - row - 1;
                        queue!(
                            writer,
                            terminal::ScrollUp(scroll_rows),
                            cursor::MoveTo(0, 0),
                        )?;
                    }

                    // No guarantee that text returned by render will not be re-layouted, so it is better to clear it.
                    queue!(writer, terminal::Clear(terminal::ClearType::FromCursorDown))?;

                    if text.contains('\n') {
                        let text = format!("{buffer}{text}");
                        let (head, tail) = split_line_tail(&text);
                        let output = render.render(head);
                        print_block(writer, &output, columns)?;
                        buffer = tail.to_string();
                    } else {
                        buffer = format!("{buffer}{text}");
                    }

                    let output = render.render_line(&buffer);
                    if output.contains('\n') {
                        let (head, tail) = split_line_tail(&output);
                        buffer_rows = print_block(writer, head, columns)?;
                        queue!(writer, style::Print(&tail),)?;

                        // No guarantee the buffer width of the buffer will not exceed the number of columns.
                        // So we calculate the number of rows needed, rather than setting it directly to 1.
                        buffer_rows += need_rows(tail, columns);
                    } else {
                        queue!(writer, style::Print(&output))?;
                        buffer_rows = need_rows(&output, columns);
                    }

                    writer.flush()?;
                }
                SseEvent::Done => {
                    if let Some(spinner) = spinner.take() {
                        spinner.stop();
                    }
                    break 'outer;
                }
            }
        }

        if poll_abort_signal(abort_signal)? {
            break;
        }
    }

    if let Some(spinner) = spinner.take() {
        spinner.stop();
    }
    Ok(())
}

/// Batches consecutive `Text` events (joined into one) within a short window,
/// same as before -- but `ReasoningStart` is never merged into that batch:
/// it's emitted as its own event, in its correct position relative to any
/// text gathered before/after it, since it's a boundary signal (see
/// `SseEvent::ReasoningStart`) a consumer needs to see distinctly, not text
/// content to concatenate.
async fn gather_events(rx: &mut UnboundedReceiver<SseEvent>) -> Vec<SseEvent> {
    let mut events: Vec<SseEvent> = vec![];
    let mut pending_text = String::new();
    let mut done = false;
    tokio::select! {
        _ = async {
            while let Some(reply_event) = rx.recv().await {
                match reply_event {
                    SseEvent::Text(v) => pending_text.push_str(&v),
                    SseEvent::ReasoningStart => {
                        if !pending_text.is_empty() {
                            events.push(SseEvent::Text(std::mem::take(&mut pending_text)));
                        }
                        events.push(SseEvent::ReasoningStart);
                    }
                    SseEvent::Done => {
                        done = true;
                        break;
                    }
                }
            }
        } => {}
        _ = tokio::time::sleep(Duration::from_millis(50)) => {}
    };
    if !pending_text.is_empty() {
        events.push(SseEvent::Text(pending_text));
    }
    if done {
        events.push(SseEvent::Done)
    }
    events
}

/// Consumes a completion's stream without displaying any of it -- used for a
/// spawned subagent's own stream when `show_subagent` is off (see
/// `render::render_stream`). The actual output/reasoning text is unaffected:
/// that's accumulated inside `SseHandler` independently of whether anything
/// here gets displayed, so the caller still gets a correct result either
/// way -- this only controls what appears on the terminal.
pub async fn drain_stream(mut rx: UnboundedReceiver<SseEvent>, abort_signal: &AbortSignal) -> Result<()> {
    loop {
        if abort_signal.aborted() {
            break;
        }
        match rx.recv().await {
            Some(SseEvent::Done) | None => break,
            Some(_) => {}
        }
    }
    Ok(())
}

fn print_block(writer: &mut Stdout, text: &str, columns: u16) -> Result<u16> {
    let mut num = 0;
    for line in text.split('\n') {
        queue!(
            writer,
            style::Print(line),
            style::Print("\n"),
            cursor::MoveLeft(columns),
        )?;
        num += 1;
    }
    Ok(num)
}

fn split_line_tail(text: &str) -> (&str, &str) {
    if let Some((head, tail)) = text.rsplit_once('\n') {
        (head, tail)
    } else {
        ("", text)
    }
}

fn need_rows(text: &str, columns: u16) -> u16 {
    let buffer_width = display_width(text).max(1) as u16;
    buffer_width.div_ceil(columns)
}

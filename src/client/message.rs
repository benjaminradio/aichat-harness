use super::Model;

use crate::function::ToolCall;
use crate::multiline_text;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Message {
    pub role: MessageRole,
    pub content: MessageContent,
    /// Present only on Assistant messages that made tool calls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Present only on Tool messages; pairs the result back to the originating call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Present only on Tool messages. Rich supplementary detail that should be
    /// logged/displayed but never sent back to the model — `content` is always the
    /// terse result the model actually sees. Generic `Value` so other tools can use
    /// it later; today only `spawn_subagent` populates it, with the subagent's full
    /// internal message trace.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trace: Option<serde_json::Value>,
    /// Present only on Assistant messages. Split out of `content` so reasoning never
    /// lives in a string a provider might re-parse or re-send — see
    /// `render::format_reasoning` for the `<think>` display-time wrapping this
    /// replaces.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

impl Message {
    pub fn new(role: MessageRole, content: MessageContent) -> Self {
        Self {
            role,
            content,
            ..Default::default()
        }
    }

    /// Build the Assistant message for one tool-calling round.
    pub fn new_assistant_tool_calls(
        text: String,
        tool_calls: Vec<ToolCall>,
        reasoning_content: Option<String>,
    ) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: MessageContent::Text(text),
            tool_calls: if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls)
            },
            reasoning_content,
            ..Default::default()
        }
    }

    /// Build one Tool-role result message, pairing back to `call_id`.
    pub fn new_tool_result(
        call_id: Option<String>,
        content: String,
        trace: Option<serde_json::Value>,
    ) -> Self {
        Self {
            role: MessageRole::Tool,
            content: MessageContent::Text(content),
            tool_call_id: call_id,
            trace,
            ..Default::default()
        }
    }

    pub fn merge_system(&mut self, system: MessageContent) {
        match (&mut self.content, system) {
            (MessageContent::Text(text), MessageContent::Text(system_text)) => {
                self.content = MessageContent::Array(vec![
                    MessageContentPart::Text { text: system_text },
                    MessageContentPart::Text {
                        text: text.to_string(),
                    },
                ])
            }
            (MessageContent::Array(list), MessageContent::Text(system_text)) => {
                list.insert(0, MessageContentPart::Text { text: system_text })
            }
            (MessageContent::Text(text), MessageContent::Array(mut system_list)) => {
                system_list.push(MessageContentPart::Text {
                    text: text.to_string(),
                });
                self.content = MessageContent::Array(system_list);
            }
            (MessageContent::Array(list), MessageContent::Array(mut system_list)) => {
                system_list.append(list);
                self.content = MessageContent::Array(system_list);
            }
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MessageRole {
    System,
    Assistant,
    #[default]
    User,
    Tool,
}

#[allow(dead_code)]
impl MessageRole {
    pub fn is_system(&self) -> bool {
        matches!(self, MessageRole::System)
    }

    pub fn is_user(&self) -> bool {
        matches!(self, MessageRole::User)
    }

    pub fn is_assistant(&self) -> bool {
        matches!(self, MessageRole::Assistant)
    }

    pub fn is_tool(&self) -> bool {
        matches!(self, MessageRole::Tool)
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Array(Vec<MessageContentPart>),
}

impl Default for MessageContent {
    fn default() -> Self {
        MessageContent::Text(String::new())
    }
}

impl MessageContent {
    pub fn render_input(
        &self,
        resolve_url_fn: impl Fn(&str) -> String,
        _agent_info: &Option<(String, Vec<String>)>,
    ) -> String {
        match self {
            MessageContent::Text(text) => multiline_text(text),
            MessageContent::Array(list) => {
                let (mut concated_text, mut files) = (String::new(), vec![]);
                for item in list {
                    match item {
                        MessageContentPart::Text { text } => {
                            concated_text = format!("{concated_text} {text}")
                        }
                        MessageContentPart::ImageUrl { image_url } => {
                            files.push(resolve_url_fn(&image_url.url))
                        }
                    }
                }
                if !concated_text.is_empty() {
                    concated_text = format!(" -- {}", multiline_text(&concated_text))
                }
                format!(".file {}{}", files.join(" "), concated_text)
            }
        }
    }

    pub fn merge_prompt(&mut self, replace_fn: impl Fn(&str) -> String) {
        match self {
            MessageContent::Text(text) => *text = replace_fn(text),
            MessageContent::Array(list) => {
                if list.is_empty() {
                    list.push(MessageContentPart::Text {
                        text: replace_fn(""),
                    })
                } else if let Some(MessageContentPart::Text { text }) = list.get_mut(0) {
                    *text = replace_fn(text)
                }
            }
        }
    }

    pub fn to_text(&self) -> String {
        match self {
            MessageContent::Text(text) => text.to_string(),
            MessageContent::Array(list) => {
                let mut parts = vec![];
                for item in list {
                    if let MessageContentPart::Text { text } = item {
                        parts.push(text.clone())
                    }
                }
                parts.join("\n\n")
            }
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MessageContentPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrl },
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ImageUrl {
    pub url: String,
}

pub fn patch_messages(messages: &mut Vec<Message>, model: &Model) {
    if messages.is_empty() {
        return;
    }
    if let Some(prefix) = model.system_prompt_prefix() {
        if messages[0].role.is_system() {
            messages[0].merge_system(MessageContent::Text(prefix.to_string()));
        } else {
            messages.insert(
                0,
                Message {
                    role: MessageRole::System,
                    content: MessageContent::Text(prefix.to_string()),
                    ..Default::default()
                },
            );
        }
    }
    if model.no_system_message() && messages[0].role.is_system() {
        let system_message = messages.remove(0);
        if let (Some(message), system) = (messages.get_mut(0), system_message.content) {
            message.merge_system(system);
        }
    }
}

pub fn extract_system_message(messages: &mut Vec<Message>) -> Option<String> {
    if messages[0].role.is_system() {
        let system_message = messages.remove(0);
        return Some(system_message.content.to_text());
    }
    None
}

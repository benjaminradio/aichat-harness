use super::input::*;
use super::*;

use crate::client::{Message, MessageContent, MessageRole};
use crate::render::MarkdownRender;

use anyhow::{bail, Context, Result};
use fancy_regex::Regex;
use inquire::{validator::Validation, Confirm, Text};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use std::fs::{read_to_string, write};
use std::path::Path;
use std::sync::LazyLock;

static RE_AUTONAME_PREFIX: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\d{8}T\d{6}-").unwrap());

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Session {
    #[serde(rename(serialize = "model", deserialize = "model"))]
    model_id: String,
    #[serde(flatten)]
    settings: RoleSettings,
    #[serde(skip_serializing_if = "Option::is_none")]
    save_session: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    compress_threshold: Option<usize>,

    #[serde(skip_serializing_if = "Option::is_none")]
    role_name: Option<String>,
    #[serde(default, skip_serializing_if = "IndexMap::is_empty")]
    agent_variables: AgentVariables,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    agent_instructions: String,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    compressed_messages: Vec<Message>,
    messages: Vec<Message>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    data_urls: HashMap<String, String>,

    #[serde(skip)]
    model: Model,
    #[serde(skip)]
    rag: Option<Arc<Rag>>,
    #[serde(skip)]
    role_prompt: String,
    #[serde(skip)]
    name: String,
    #[serde(skip)]
    path: Option<String>,
    #[serde(skip)]
    dirty: bool,
    #[serde(skip)]
    save_session_this_time: bool,
    #[serde(skip)]
    compressing: bool,
    #[serde(skip)]
    autoname: Option<AutoName>,
    #[serde(skip)]
    tokens: usize,
}

impl Session {
    pub fn new(config: &Config, name: &str) -> Self {
        let role = config.extract_role();
        let mut session = Self {
            name: name.to_string(),
            save_session: config.save_session,
            ..Default::default()
        };
        session.set_role(role);
        session.dirty = false;
        session
    }

    pub fn load(config: &Config, name: &str, path: &Path) -> Result<Self> {
        let content = read_to_string(path)
            .with_context(|| format!("Failed to load session {} at {}", name, path.display()))?;
        let mut session: Self =
            serde_yaml::from_str(&content).with_context(|| format!("Invalid session {name}"))?;

        session.model = Model::retrieve_model(config, &session.model_id, ModelType::Chat)?;

        if let Some(autoname) = name.strip_prefix("_/") {
            session.name = TEMP_SESSION_NAME.to_string();
            session.path = None;
            if let Ok(true) = RE_AUTONAME_PREFIX.is_match(autoname) {
                session.autoname = Some(AutoName::new(autoname[16..].to_string()));
            }
        } else {
            session.name = name.to_string();
            session.path = Some(path.display().to_string());
        }

        if let Some(role_name) = &session.role_name {
            if let Ok(role) = config.retrieve_role(role_name) {
                session.role_prompt = role.prompt().to_string();
            }
        }

        session.update_tokens();

        Ok(session)
    }

    pub fn is_empty(&self) -> bool {
        self.messages.is_empty() && self.compressed_messages.is_empty()
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn role_name(&self) -> Option<&str> {
        self.role_name.as_deref()
    }

    pub fn dirty(&self) -> bool {
        self.dirty
    }

    pub fn save_session(&self) -> Option<bool> {
        self.save_session
    }

    pub fn tokens(&self) -> usize {
        self.tokens
    }

    pub fn update_tokens(&mut self) {
        self.tokens = self.model().total_tokens(&self.messages);
    }

    pub fn has_user_messages(&self) -> bool {
        self.messages.iter().any(|v| v.role.is_user())
    }

    pub fn user_messages_len(&self) -> usize {
        self.messages.iter().filter(|v| v.role.is_user()).count()
    }

    pub fn export(&self) -> Result<String> {
        let mut data = json!({
            "path": self.path,
            "model": self.model().id(),
        });
        if let Some(temperature) = self.temperature() {
            data["temperature"] = temperature.into();
        }
        if let Some(top_p) = self.top_p() {
            data["top_p"] = top_p.into();
        }
        if let Some(use_tools) = self.use_tools() {
            data["use_tools"] = use_tools.into();
        }
        if let Some(save_session) = self.save_session() {
            data["save_session"] = save_session.into();
        }
        let (tokens, percent) = self.tokens_usage();
        data["total_tokens"] = tokens.into();
        if let Some(max_input_tokens) = self.model().max_input_tokens() {
            data["max_input_tokens"] = max_input_tokens.into();
        }
        if percent != 0.0 {
            data["total/max"] = format!("{percent}%").into();
        }
        data["messages"] = json!(self.messages);

        let output = serde_yaml::to_string(&data)
            .with_context(|| format!("Unable to show info about session '{}'", &self.name))?;
        Ok(output)
    }

    pub fn render(
        &self,
        render: &mut MarkdownRender,
        agent_info: &Option<(String, Vec<String>)>,
        show_thinking: bool,
        show_subagent: bool,
    ) -> Result<String> {
        let mut items = vec![];

        if let Some(path) = &self.path {
            items.push(("path", path.to_string()));
        }

        if let Some(autoname) = self.autoname() {
            items.push(("autoname", autoname.to_string()));
        }

        items.push(("model", self.model().id()));

        if let Some(temperature) = self.temperature() {
            items.push(("temperature", temperature.to_string()));
        }
        if let Some(top_p) = self.top_p() {
            items.push(("top_p", top_p.to_string()));
        }

        if let Some(use_tools) = self.use_tools() {
            items.push(("use_tools", use_tools));
        }

        if let Some(save_session) = self.save_session() {
            items.push(("save_session", save_session.to_string()));
        }

        if let Some(compress_threshold) = self.compress_threshold {
            items.push(("compress_threshold", compress_threshold.to_string()));
        }

        if let Some(max_input_tokens) = self.model().max_input_tokens() {
            items.push(("max_input_tokens", max_input_tokens.to_string()));
        }

        let mut lines: Vec<String> = items
            .iter()
            .map(|(name, value)| format!("{name:<20}{value}"))
            .collect();

        lines.push(String::new());

        if !self.is_empty() {
            let resolve_url_fn = |url: &str| resolve_data_url(&self.data_urls, url.to_string());

            for message in &self.messages {
                match message.role {
                    MessageRole::System => {
                        lines.push(
                            render
                                .render(&message.content.render_input(resolve_url_fn, agent_info)),
                        );
                    }
                    MessageRole::Assistant => {
                        if show_thinking {
                            if let Some(reasoning) = &message.reasoning_content {
                                let text = format_reasoning(reasoning);
                                if !text.is_empty() {
                                    lines.push(render.render(&text));
                                }
                            }
                        }
                        if let MessageContent::Text(text) = &message.content {
                            if !text.is_empty() {
                                lines.push(render.render(text));
                            }
                        }
                        if let Some(tool_calls) = &message.tool_calls {
                            for call in tool_calls {
                                let name = match agent_info {
                                    Some((agent_name, functions))
                                        if functions.contains(&call.name) =>
                                    {
                                        format!("{agent_name}-{}", call.name)
                                    }
                                    _ => call.name.clone(),
                                };
                                lines.push(dimmed_text(&format!(
                                    "Call {name} {}",
                                    call.arguments
                                )));
                            }
                        }
                        lines.push("".into());
                    }
                    MessageRole::User => {
                        lines.push(format!(
                            ">> {}",
                            message.content.render_input(resolve_url_fn, agent_info)
                        ));
                    }
                    MessageRole::Tool => {
                        let text = message.content.to_text();
                        let text = if show_subagent {
                            format_subagent_trace(message.trace.as_ref(), &text)
                        } else {
                            text
                        };
                        lines.push(text);
                    }
                }
            }
        }

        Ok(lines.join("\n"))
    }

    pub fn tokens_usage(&self) -> (usize, f32) {
        let tokens = self.tokens();
        let max_input_tokens = self.model().max_input_tokens().unwrap_or_default();
        let percent = if max_input_tokens == 0 {
            0.0
        } else {
            let percent = tokens as f32 / max_input_tokens as f32 * 100.0;
            (percent * 100.0).round() / 100.0
        };
        (tokens, percent)
    }

    pub fn set_role(&mut self, role: Role) {
        self.model_id = role.model().id();
        self.settings.temperature = role.temperature();
        self.settings.top_p = role.top_p();
        self.settings.use_tools = role.use_tools();
        self.settings.use_agents = role.use_agents();
        self.settings.max_spawn_depth = role.max_spawn_depth();
        self.settings.max_subagent_turns = role.max_subagent_turns();
        self.model = role.model().clone();
        self.rag = role.rag();
        self.role_name = convert_option_string(role.name());
        self.role_prompt = role.prompt().to_string();
        self.dirty = true;
        self.update_tokens();
    }

    pub fn clear_role(&mut self) {
        self.role_name = None;
        self.role_prompt.clear();
        self.rag = None;
    }

    pub fn sync_agent(&mut self, agent: &Role) {
        self.role_name = None;
        self.role_prompt = agent.interpolated_instructions();
        self.agent_variables = agent.variables().clone();
        self.agent_instructions = self.role_prompt.clone();
        self.rag = agent.rag();
    }

    pub fn agent_variables(&self) -> &AgentVariables {
        &self.agent_variables
    }

    pub fn agent_instructions(&self) -> &str {
        &self.agent_instructions
    }

    /// This session's stored message history -- `messages`, not the (larger,
    /// including `input`'s own about-to-be-added turn) result of
    /// `build_messages`. Used to detect whether the session was left with an
    /// unanswered `MessageRole::User` turn (e.g. the process was interrupted
    /// before the assistant replied) -- see
    /// `crate::repl::maybe_complete_pending_turn`.
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    pub fn set_save_session(&mut self, value: Option<bool>) {
        if self.save_session != value {
            self.save_session = value;
            self.dirty = true;
        }
    }

    pub fn set_save_session_this_time(&mut self) {
        self.save_session_this_time = true;
    }

    pub fn set_compress_threshold(&mut self, value: Option<usize>) {
        if self.compress_threshold != value {
            self.compress_threshold = value;
            self.dirty = true;
        }
    }

    pub fn need_compress(&self, global_compress_threshold: usize) -> bool {
        if self.compressing {
            return false;
        }
        let threshold = self.compress_threshold.unwrap_or(global_compress_threshold);
        if threshold < 1 {
            return false;
        }
        self.tokens() > threshold
    }

    pub fn compressing(&self) -> bool {
        self.compressing
    }

    pub fn set_compressing(&mut self, compressing: bool) {
        self.compressing = compressing;
    }

    pub fn compress(&mut self, mut prompt: String) {
        if let Some(system_prompt) = self.messages.first().and_then(|v| {
            if MessageRole::System == v.role {
                let content = v.content.to_text();
                if !content.is_empty() {
                    return Some(content);
                }
            }
            None
        }) {
            prompt = format!("{system_prompt}\n\n{prompt}",);
        }
        self.compressed_messages.append(&mut self.messages);
        self.messages.push(Message::new(
            MessageRole::System,
            MessageContent::Text(prompt),
        ));
        self.dirty = true;
        self.update_tokens();
    }

    pub fn need_autoname(&self) -> bool {
        self.autoname.as_ref().map(|v| v.need()).unwrap_or_default()
    }

    pub fn set_autonaming(&mut self, naming: bool) {
        if let Some(v) = self.autoname.as_mut() {
            v.naming = naming;
        }
    }

    pub fn chat_history_for_autonaming(&self) -> Option<String> {
        self.autoname.as_ref().and_then(|v| v.chat_history.clone())
    }

    pub fn autoname(&self) -> Option<&str> {
        self.autoname.as_ref().and_then(|v| v.name.as_deref())
    }

    pub fn set_autoname(&mut self, value: &str) {
        let name = value
            .chars()
            .map(|v| if v.is_alphanumeric() { v } else { '-' })
            .collect();
        self.autoname = Some(AutoName::new(name));
    }

    pub fn exit(&mut self, session_dir: &Path, is_repl: bool) -> Result<()> {
        let mut save_session = self.save_session();
        if self.save_session_this_time {
            save_session = Some(true);
        }
        if self.dirty && save_session != Some(false) {
            let mut session_dir = session_dir.to_path_buf();
            let mut session_name = self.name().to_string();
            if save_session.is_none() {
                if !is_repl {
                    return Ok(());
                }
                let ans = Confirm::new("Save session?").with_default(false).prompt()?;
                if !ans {
                    return Ok(());
                }
                if session_name == TEMP_SESSION_NAME {
                    session_name = Text::new("Session name:")
                        .with_validator(|input: &str| {
                            let input = input.trim();
                            if input.is_empty() {
                                Ok(Validation::Invalid("This name is required".into()))
                            } else if input == TEMP_SESSION_NAME {
                                Ok(Validation::Invalid("This name is reserved".into()))
                            } else {
                                Ok(Validation::Valid)
                            }
                        })
                        .prompt()?;
                }
            } else if save_session == Some(true) && session_name == TEMP_SESSION_NAME {
                session_dir = session_dir.join("_");
                ensure_parent_exists(&session_dir).with_context(|| {
                    format!("Failed to create directory '{}'", session_dir.display())
                })?;

                let now = chrono::Local::now();
                session_name = now.format("%Y%m%dT%H%M%S").to_string();
                if let Some(autoname) = self.autoname() {
                    session_name = format!("{session_name}-{autoname}")
                }
            }
            let session_path = session_dir.join(format!("{session_name}.yaml"));
            self.save(&session_name, &session_path, is_repl)?;
        }
        Ok(())
    }

    pub fn save(&mut self, session_name: &str, session_path: &Path, is_repl: bool) -> Result<()> {
        ensure_parent_exists(session_path)?;

        self.path = Some(session_path.display().to_string());

        let content = serde_yaml::to_string(&self)
            .with_context(|| format!("Failed to serde session '{}'", self.name))?;
        write(session_path, content).with_context(|| {
            format!(
                "Failed to write session '{}' to '{}'",
                self.name,
                session_path.display()
            )
        })?;

        if is_repl {
            println!("✓ Saved the session to '{}'.", session_path.display());
        }

        if self.name() != session_name {
            self.name = session_name.to_string()
        }

        self.dirty = false;

        Ok(())
    }

    pub fn guard_empty(&self) -> Result<()> {
        if !self.is_empty() {
            bail!("Cannot perform this operation because the session has messages, please `.empty session` first.");
        }
        Ok(())
    }

    pub fn add_message(
        &mut self,
        input: &Input,
        output: &str,
        reasoning_content: Option<String>,
    ) -> Result<()> {
        if input.continue_output().is_some() {
            if let Some(message) = self.messages.last_mut() {
                if let MessageContent::Text(text) = &mut message.content {
                    *text = format!("{text}{output}");
                }
                if reasoning_content.is_some() {
                    message.reasoning_content = reasoning_content;
                }
            }
        } else if input.regenerate() {
            if let Some(message) = self.messages.last_mut() {
                if let MessageContent::Text(text) = &mut message.content {
                    *text = output.to_string();
                }
                message.reasoning_content = reasoning_content;
            }
        } else {
            if self.messages.is_empty() {
                if self.name == TEMP_SESSION_NAME && self.save_session == Some(true) {
                    let raw_input = input.raw();
                    let chat_history = format!("USER: {raw_input}\nASSISTANT: {output}\n");
                    self.autoname = Some(AutoName::new_from_chat_history(chat_history));
                }
                self.messages.extend(input.role().build_messages(input));
            } else {
                self.messages
                    .push(Message::new(MessageRole::User, input.message_content()));
            }
            self.data_urls.extend(input.data_urls());
            self.messages.extend(input.pending_messages().iter().cloned());
            let mut message = Message::new(MessageRole::Assistant, MessageContent::Text(output.to_string()));
            message.reasoning_content = reasoning_content;
            self.messages.push(message);
        }
        self.dirty = true;
        self.update_tokens();
        Ok(())
    }

    pub fn clear_messages(&mut self) {
        self.messages.clear();
        self.compressed_messages.clear();
        self.data_urls.clear();
        self.autoname = None;
        self.dirty = true;
        self.update_tokens();
    }

    pub fn echo_messages(&self, input: &Input) -> String {
        let messages = self.build_messages(input);
        serde_yaml::to_string(&messages).unwrap_or_else(|_| "Unable to echo message".into())
    }

    /// The last message in the returned list is `MessageRole::User` in every
    /// case except `continue_output`/`regenerate` -- including when this
    /// session's own stored history already ends in an unanswered User turn
    /// (most commonly: resuming a session that was interrupted before the
    /// assistant replied) and `input` carries no new content: that existing
    /// dangling turn is used as-is rather than piling an empty turn on top
    /// of it. Mirrors `Role::build_messages`'s handling of a prompt
    /// template's own trailing, unmatched `### INPUT:` section -- which is
    /// exactly what populates a brand new session's initial messages here
    /// (the `len == 0` branch below), so the two compose correctly.
    pub fn build_messages(&self, input: &Input) -> Vec<Message> {
        let mut messages = self.messages.clone();
        if input.continue_output().is_some() {
            return messages;
        } else if input.regenerate() {
            while let Some(last) = messages.last() {
                if !last.role.is_user() {
                    messages.pop();
                } else {
                    break;
                }
            }
            return messages;
        }
        let len = messages.len();
        if len == 0 {
            messages = input.role().build_messages(input);
        } else {
            if len == 1 && self.compressed_messages.len() >= 2 {
                if let Some(index) = self
                    .compressed_messages
                    .iter()
                    .rposition(|v| v.role == MessageRole::User)
                {
                    messages.extend(self.compressed_messages[index..].to_vec());
                }
            }
            let already_ends_in_user = messages.last().is_some_and(|m| m.role.is_user());
            if !input.is_empty() || !already_ends_in_user {
                messages.push(Message::new(MessageRole::User, input.message_content()));
            }
        }
        messages
    }
}

impl RoleLike for Session {
    fn to_role(&self) -> Role {
        let role_name = self.role_name.as_deref().unwrap_or_default();
        let mut role = Role::new(role_name, &self.role_prompt);
        role.sync(self);
        role.rag = self.rag.clone();
        role
    }

    fn model(&self) -> &Model {
        &self.model
    }

    fn temperature(&self) -> Option<f64> {
        self.settings.temperature
    }

    fn top_p(&self) -> Option<f64> {
        self.settings.top_p
    }

    fn use_tools(&self) -> Option<String> {
        self.settings.use_tools.clone()
    }

    fn use_agents(&self) -> Option<String> {
        self.settings.use_agents.clone()
    }

    fn max_spawn_depth(&self) -> Option<usize> {
        self.settings.max_spawn_depth
    }

    fn max_subagent_turns(&self) -> Option<usize> {
        self.settings.max_subagent_turns
    }

    fn rag(&self) -> Option<Arc<Rag>> {
        self.rag.clone()
    }

    fn set_model(&mut self, model: Model) {
        if self.model().id() != model.id() {
            self.model_id = model.id();
            self.model = model;
            self.dirty = true;
            self.update_tokens();
        }
    }

    fn set_temperature(&mut self, value: Option<f64>) {
        if self.settings.temperature != value {
            self.settings.temperature = value;
            self.dirty = true;
        }
    }

    fn set_top_p(&mut self, value: Option<f64>) {
        if self.settings.top_p != value {
            self.settings.top_p = value;
            self.dirty = true;
        }
    }

    fn set_use_tools(&mut self, value: Option<String>) {
        if self.settings.use_tools != value {
            self.settings.use_tools = value;
            self.dirty = true;
        }
    }

    fn set_use_agents(&mut self, value: Option<String>) {
        if self.settings.use_agents != value {
            self.settings.use_agents = value;
            self.dirty = true;
        }
    }

    fn set_max_spawn_depth(&mut self, value: Option<usize>) {
        if self.settings.max_spawn_depth != value {
            self.settings.max_spawn_depth = value;
            self.dirty = true;
        }
    }

    fn set_max_subagent_turns(&mut self, value: Option<usize>) {
        if self.settings.max_subagent_turns != value {
            self.settings.max_subagent_turns = value;
            self.dirty = true;
        }
    }

    fn set_rag(&mut self, value: Option<Arc<Rag>>) {
        self.rag = value;
        self.dirty = true;
    }
}

#[derive(Debug, Clone, Default)]
struct AutoName {
    naming: bool,
    chat_history: Option<String>,
    name: Option<String>,
}

impl AutoName {
    pub fn new(name: String) -> Self {
        Self {
            name: Some(name),
            ..Default::default()
        }
    }
    pub fn new_from_chat_history(chat_history: String) -> Self {
        Self {
            chat_history: Some(chat_history),
            ..Default::default()
        }
    }
    pub fn need(&self) -> bool {
        !self.naming && self.chat_history.is_some() && self.name.is_none()
    }
}

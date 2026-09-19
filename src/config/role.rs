use super::*;

use crate::client::{Message, MessageContent, MessageRole, Model};
use crate::function::{run_llm_function, Functions};

use anyhow::{Context, Result};
use inquire::{validator::Validation, Confirm, Text};
use rust_embed::Embed;
use serde::{Deserialize, Serialize};
use std::fs::read_to_string;

pub const SHELL_ROLE: &str = "%shell%";
pub const EXPLAIN_SHELL_ROLE: &str = "%explain-shell%";
pub const CODE_ROLE: &str = "%code%";
pub const CREATE_TITLE_ROLE: &str = "%create-title%";

pub const INPUT_PLACEHOLDER: &str = "__INPUT__";

const DEFAULT_AGENT_RAG_NAME: &str = "rag";

#[derive(Embed)]
#[folder = "assets/roles/"]
struct RolesAsset;

pub trait RoleLike {
    fn to_role(&self) -> Role;
    fn model(&self) -> &Model;
    fn temperature(&self) -> Option<f64>;
    fn top_p(&self) -> Option<f64>;
    fn use_tools(&self) -> Option<String>;
    fn use_agents(&self) -> Option<String>;
    fn max_spawn_depth(&self) -> Option<usize>;
    fn max_subagent_turns(&self) -> Option<usize>;
    fn rag(&self) -> Option<Arc<Rag>>;
    fn set_model(&mut self, model: Model);
    fn set_temperature(&mut self, value: Option<f64>);
    fn set_top_p(&mut self, value: Option<f64>);
    fn set_use_tools(&mut self, value: Option<String>);
    fn set_use_agents(&mut self, value: Option<String>);
    fn set_max_spawn_depth(&mut self, value: Option<usize>);
    fn set_max_subagent_turns(&mut self, value: Option<usize>);
    fn set_rag(&mut self, value: Option<Arc<Rag>>);
}

/// Shared settings embedded (by value) in `Role`, `Session`, and `Config`'s own
/// top-level global-default fields. Introduced so that adding a new per-role/session/
/// global setting is a one-line change to this struct instead of a hand-rolled
/// enumeration across every implementor of `RoleLike`.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct RoleSettings {
    #[serde(rename = "model", skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub use_tools: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub use_agents: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_spawn_depth: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_subagent_turns: Option<usize>,
}

impl RoleSettings {
    /// Overlay `other`'s non-None fields onto self.
    pub fn merge_from(&mut self, other: &RoleSettings) {
        macro_rules! take {
            ($f:ident) => {
                if other.$f.is_some() {
                    self.$f = other.$f.clone();
                }
            };
        }
        take!(model_id);
        take!(temperature);
        take!(top_p);
        take!(use_tools);
        take!(use_agents);
        take!(max_spawn_depth);
        take!(max_subagent_turns);
    }
}

/// Controls how a missing (no supplied value, no `default:`) required agent variable
/// is handled during variable resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VariableInitMode {
    /// Prompt for the value on a TTY. Caller is responsible for only selecting this
    /// mode when a TTY is actually available (mirrors today's `IS_STDOUT_TERMINAL`
    /// gate, moved to the call site so this function has no branching per mode).
    Interactive,
    /// Bail with a "required variables" error listing every unset variable.
    FailOnMissing,
    /// Explicitly substitute an empty string for the variable rather than omitting
    /// the key — omitting it would leave the literal `{{key}}` placeholder
    /// un-substituted in `interpolated_instructions()`'s output. Used by non-interactive/
    /// info-flag activation and by subagent spawning (§7).
    DefaultEmpty,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
pub enum RoleKind {
    #[default]
    Role,
    Agent,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Role {
    #[serde(skip)]
    pub kind: RoleKind,
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub version: String,
    /// Raw text for `kind == Role`; `{{variable}}` / `__INPUT__` template for
    /// `kind == Agent`. Use `effective_prompt()` for message construction, not this
    /// field directly, when the value might come from either kind.
    #[serde(default)]
    pub prompt: String,
    /// Agent-only; always `false` and inert for `kind == Role`.
    #[serde(default)]
    pub dynamic_instructions: bool,
    #[serde(flatten)]
    pub settings: RoleSettings,
    /// Agent-only; always empty/inert for `kind == Role`.
    #[serde(default)]
    pub variables: Vec<AgentVariable>,
    /// Agent-only; always empty/inert for `kind == Role`.
    #[serde(default)]
    pub conversation_starters: Vec<String>,
    /// Agent-only; always empty/inert for `kind == Role`.
    #[serde(default)]
    pub documents: Vec<String>,

    // Runtime-only fields below, never (de)serialized from the definition file.
    #[serde(skip)]
    pub rag: Option<Arc<Rag>>,
    /// Always `Functions::default()` for `kind == Role`.
    #[serde(skip)]
    pub functions: Functions,
    #[serde(skip)]
    shared_variables: AgentVariables,
    #[serde(skip)]
    session_variables: Option<AgentVariables>,
    #[serde(skip)]
    shared_dynamic_instructions: Option<String>,
    #[serde(skip)]
    session_dynamic_instructions: Option<String>,
    #[serde(skip)]
    model: Model,
}

impl Role {
    /// Build an ephemeral role directly from a prompt string — used for ad-hoc
    /// contexts like `.prompt <text>` and `.rag` with no role/agent active, not for
    /// loading a `roles/<name>.yaml` file (see `load_role` for that).
    pub fn new(name: &str, prompt: &str) -> Self {
        let mut prompt = prompt.to_string();
        interpolate_variables(&mut prompt);
        Self {
            name: name.to_string(),
            prompt,
            ..Default::default()
        }
    }

    /// Load a `roles/<name>.yaml` file. This is the only loader for user-defined
    /// roles — the old `.md` + frontmatter format is gone entirely.
    pub fn load_role(name: &str) -> Result<Self> {
        let path = Config::role_file(name);
        let content = read_to_string(&path)
            .with_context(|| format!("Failed to read role file at '{}'", path.display()))?;
        let mut role: Role = serde_yaml::from_str(&content)
            .with_context(|| format!("Failed to load role at '{}'", path.display()))?;
        role.kind = RoleKind::Role;
        role.name = name.to_string();
        interpolate_variables(&mut role.prompt);
        Ok(role)
    }

    /// Load `agents/<name>/{index.yaml,functions.json,<rag>.yaml}`. This replaces
    /// today's split between the shareable `functions_dir()/agents/<name>/` tree and
    /// the local `local_path("agents")/<name>/config.yaml` override — there is no
    /// override file anymore, `index.yaml` is the only settings/definition source.
    ///
    /// `force_init_rag`: when `true`, skip the interactive "init RAG?" confirmation
    /// and build the index unconditionally if `documents` is non-empty and no cache
    /// exists yet. Normal agent activation passes `false` (preserves today's
    /// prompt-or-skip behavior exactly); subagent spawning (§7) passes `true`.
    pub async fn load_agent(
        config: &GlobalConfig,
        name: &str,
        force_init_rag: bool,
        abort_signal: AbortSignal,
    ) -> Result<Self> {
        let dir = Config::agent_dir(name);
        let index_path = dir.join("index.yaml");
        if !index_path.exists() {
            bail!("Unknown agent `{name}`");
        }
        let mut role: Role = serde_yaml::from_str(&read_to_string(&index_path)?)
            .with_context(|| format!("Failed to load agent at '{}'", index_path.display()))?;
        role.kind = RoleKind::Agent;
        role.name = name.to_string();

        let functions_path = dir.join("functions.json");
        role.functions = if functions_path.exists() {
            Functions::init(&functions_path)?
        } else {
            Functions::default()
        };
        role.replace_tools_placeholder();

        role.load_envs();

        let model = {
            let cfg = config.read();
            match role.settings.model_id.clone() {
                Some(model_id) => Model::retrieve_model(&cfg, &model_id, ModelType::Chat)?,
                None => {
                    if role.settings.temperature.is_none() {
                        role.settings.temperature = cfg.settings.temperature;
                    }
                    if role.settings.top_p.is_none() {
                        role.settings.top_p = cfg.settings.top_p;
                    }
                    cfg.current_model().clone()
                }
            }
        };
        role.model = model;

        let rag_path = Config::agent_rag_file(name, DEFAULT_AGENT_RAG_NAME);
        let rag = if rag_path.exists() {
            Some(Arc::new(Rag::load(config, DEFAULT_AGENT_RAG_NAME, &rag_path)?))
        } else if !role.documents.is_empty() && !config.read().info_flag {
            let mut ans = force_init_rag;
            if !force_init_rag && *IS_STDOUT_TERMINAL {
                ans = Confirm::new("The agent has the documents, init RAG?")
                    .with_default(true)
                    .prompt()?;
            }
            if ans {
                let mut document_paths = vec![];
                for path in &role.documents {
                    if is_url(path) {
                        document_paths.push(path.to_string());
                    } else {
                        let new_path = safe_join_path(&dir, path)
                            .ok_or_else(|| anyhow!("Invalid document path: '{path}'"))?;
                        document_paths.push(new_path.display().to_string())
                    }
                }
                let rag = Rag::init(
                    config,
                    DEFAULT_AGENT_RAG_NAME,
                    &rag_path,
                    &document_paths,
                    abort_signal,
                )
                .await?;
                Some(Arc::new(rag))
            } else {
                None
            }
        } else {
            None
        };
        role.rag = rag;

        Ok(role)
    }

    fn load_envs(&mut self) {
        let name = self.name.clone();
        let with_prefix = |v: &str| normalize_env_name(&format!("{name}_{v}"));

        if let Some(v) = read_env_value::<String>(&with_prefix("model")) {
            self.settings.model_id = v;
        }
        if let Some(v) = read_env_value::<f64>(&with_prefix("temperature")) {
            self.settings.temperature = v;
        }
        if let Some(v) = read_env_value::<f64>(&with_prefix("top_p")) {
            self.settings.top_p = v;
        }
        if let Some(v) = read_env_value::<String>(&with_prefix("use_tools")) {
            self.settings.use_tools = v;
        }
        if let Some(v) = read_env_value::<String>(&with_prefix("use_agents")) {
            self.settings.use_agents = v;
        }
        if let Some(v) = read_env_value::<usize>(&with_prefix("max_spawn_depth")) {
            self.settings.max_spawn_depth = v;
        }
        if let Some(v) = read_env_value::<usize>(&with_prefix("max_subagent_turns")) {
            self.settings.max_subagent_turns = v;
        }
        if let Ok(v) = env::var(with_prefix("variables")) {
            if let Ok(v) = serde_json::from_str(&v) {
                self.shared_variables = v;
            }
        }
    }

    pub fn builtin(name: &str) -> Result<Self> {
        let content = RolesAsset::get(&format!("{name}.yaml"))
            .ok_or_else(|| anyhow!("Unknown role `{name}`"))?;
        let content = unsafe { std::str::from_utf8_unchecked(&content.data) };
        let mut role: Role = serde_yaml::from_str(content)
            .with_context(|| format!("Failed to load role `{name}`"))?;
        role.kind = RoleKind::Role;
        role.name = name.to_string();
        interpolate_variables(&mut role.prompt);
        Ok(role)
    }

    pub fn list_builtin_role_names() -> Vec<String> {
        RolesAsset::iter()
            .filter_map(|v| v.strip_suffix(".yaml").map(|v| v.to_string()))
            .collect()
    }

    pub fn list_builtin_roles() -> Vec<Self> {
        RolesAsset::iter()
            .filter_map(|v| v.strip_suffix(".yaml").map(|v| v.to_string()))
            .filter_map(|v| Role::builtin(&v).ok())
            .collect()
    }

    pub fn has_args(&self) -> bool {
        self.name.contains('#')
    }

    /// Full YAML serialization of this `Role`/`Agent` — used both to write it back to
    /// disk (`.role save` / `.agent save`) and for `.info role` / `.info agent`
    /// display. Runtime-only fields (`rag`, `functions`, variable state, `model`,
    /// `kind`) are `#[serde(skip)]` and never appear in the output.
    pub fn export(&self) -> Result<String> {
        let data = serde_yaml::to_string(self)?;
        Ok(data)
    }

    pub fn save(&mut self, new_name: &str, path: &Path, is_repl: bool) -> Result<()> {
        ensure_parent_exists(path)?;

        if new_name != self.name {
            self.name = new_name.to_string();
        }

        let content = self.export()?;
        std::fs::write(path, content)
            .with_context(|| format!("Failed to write '{}' to {}", self.name, path.display()))?;

        if is_repl {
            println!("✓ Saved to '{}'.", path.display());
        }

        Ok(())
    }

    pub fn sync<T: RoleLike>(&mut self, role_like: &T) {
        self.set_model(role_like.model().clone());
        let mut settings = RoleSettings::default();
        settings.temperature = role_like.temperature();
        settings.top_p = role_like.top_p();
        settings.use_tools = role_like.use_tools();
        settings.use_agents = role_like.use_agents();
        settings.max_spawn_depth = role_like.max_spawn_depth();
        settings.max_subagent_turns = role_like.max_subagent_turns();
        self.settings.merge_from(&settings);
    }

    pub fn is_derived(&self) -> bool {
        self.name.is_empty()
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn model_id(&self) -> Option<&str> {
        self.settings.model_id.as_deref()
    }

    /// Raw prompt text for `kind == Role`; the fully-interpolated instructions for
    /// `kind == Agent`. This is the one place `kind` actively dispatches rather than
    /// just gating already-empty data — an agent's effective prompt is a
    /// *computation* over `prompt` + `variables` + `dynamic_instructions`, not a peer
    /// value the way a role's prompt is.
    pub fn effective_prompt(&self) -> String {
        match self.kind {
            RoleKind::Role => self.prompt.clone(),
            RoleKind::Agent => self.interpolated_instructions(),
        }
    }

    /// Kept as a thin alias so existing call sites that read a role's raw prompt text
    /// don't need to know about `kind` for the `kind == Role` case; callers dealing
    /// with either kind should prefer `effective_prompt()`.
    pub fn prompt(&self) -> &str {
        &self.prompt
    }

    pub fn is_empty_prompt(&self) -> bool {
        self.effective_prompt().is_empty()
    }

    pub fn is_embedded_prompt(&self) -> bool {
        self.effective_prompt().contains(INPUT_PLACEHOLDER)
    }

    pub fn echo_messages(&self, input: &Input) -> String {
        let prompt = self.effective_prompt();
        let input_markdown = input.render();
        if self.is_empty_prompt() {
            input_markdown
        } else if self.is_embedded_prompt() {
            prompt.replace(INPUT_PLACEHOLDER, &input_markdown)
        } else {
            format!("{prompt}\n\n{input_markdown}")
        }
    }

    pub fn build_messages(&self, input: &Input) -> Vec<Message> {
        let prompt = self.effective_prompt();
        let mut content = input.message_content();
        let mut messages = if self.is_empty_prompt() {
            vec![Message::new(MessageRole::User, content)]
        } else if self.is_embedded_prompt() {
            content.merge_prompt(|v: &str| prompt.replace(INPUT_PLACEHOLDER, v));
            vec![Message::new(MessageRole::User, content)]
        } else {
            let mut messages = vec![];
            let (system, cases) = parse_structure_prompt(&prompt);
            if !system.is_empty() {
                messages.push(Message::new(
                    MessageRole::System,
                    MessageContent::Text(system.to_string()),
                ));
            }
            if !cases.is_empty() {
                messages.extend(cases.into_iter().flat_map(|(i, o)| {
                    vec![
                        Message::new(MessageRole::User, MessageContent::Text(i.to_string())),
                        Message::new(MessageRole::Assistant, MessageContent::Text(o.to_string())),
                    ]
                }));
            }
            messages.push(Message::new(MessageRole::User, content));
            messages
        };
        if let Some(text) = input.continue_output() {
            messages.push(Message::new(
                MessageRole::Assistant,
                MessageContent::Text(text.into()),
            ));
        }
        messages
    }

    // ---- Ported from the old `Agent` type. Callable regardless of `kind`, but only
    // ---- meaningfully do something for `kind == Agent` — e.g. `is_dynamic_instructions()`
    // ---- just returns `self.dynamic_instructions`, which is always `false` for a
    // ---- `Role`-kind instance because nothing ever sets it true for one.

    pub fn banner(&self) -> String {
        let starters = if self.conversation_starters.is_empty() {
            String::new()
        } else {
            let starters = self
                .conversation_starters
                .iter()
                .map(|v| format!("- {v}"))
                .collect::<Vec<_>>()
                .join("\n");
            format!(
                r#"

## Conversation Starters
{starters}"#
            )
        };
        format!(
            "# {} {}\n{}{starters}",
            self.name, self.version, self.description
        )
    }

    pub fn functions(&self) -> &Functions {
        &self.functions
    }

    pub fn rag(&self) -> Option<Arc<Rag>> {
        self.rag.clone()
    }

    pub fn set_rag(&mut self, rag: Option<Arc<Rag>>) {
        self.rag = rag;
    }

    pub fn conversation_staters(&self) -> &[String] {
        &self.conversation_starters
    }

    pub fn interpolated_instructions(&self) -> String {
        let mut output = self
            .session_dynamic_instructions
            .clone()
            .or_else(|| self.shared_dynamic_instructions.clone())
            .unwrap_or_else(|| self.prompt.clone());
        for (k, v) in self.variables() {
            output = output.replace(&format!("{{{{{k}}}}}"), v)
        }
        interpolate_variables(&mut output);
        output
    }

    pub fn variables(&self) -> &AgentVariables {
        match &self.session_variables {
            Some(variables) => variables,
            None => &self.shared_variables,
        }
    }

    pub fn variable_envs(&self) -> HashMap<String, String> {
        self.variables()
            .iter()
            .map(|(k, v)| {
                (
                    format!("LLM_AGENT_VAR_{}", normalize_env_name(k)),
                    v.clone(),
                )
            })
            .collect()
    }

    pub fn shared_variables(&self) -> &AgentVariables {
        &self.shared_variables
    }

    pub fn set_shared_variables(&mut self, shared_variables: AgentVariables) {
        self.shared_variables = shared_variables;
    }

    pub fn set_session_variables(&mut self, session_variables: AgentVariables) {
        self.session_variables = Some(session_variables);
    }

    pub fn defined_variables(&self) -> &[AgentVariable] {
        &self.variables
    }

    pub fn exit_session(&mut self) {
        self.session_variables = None;
        self.session_dynamic_instructions = None;
    }

    pub fn is_dynamic_instructions(&self) -> bool {
        self.dynamic_instructions
    }

    pub fn update_shared_dynamic_instructions(&mut self, force: bool) -> Result<()> {
        if self.is_dynamic_instructions() && (force || self.shared_dynamic_instructions.is_none()) {
            self.shared_dynamic_instructions = Some(self.run_instructions_fn()?);
        }
        Ok(())
    }

    pub fn update_session_dynamic_instructions(&mut self, value: Option<String>) -> Result<()> {
        if self.is_dynamic_instructions() {
            let value = match value {
                Some(v) => v,
                None => self.run_instructions_fn()?,
            };
            self.session_dynamic_instructions = Some(value);
        }
        Ok(())
    }

    fn run_instructions_fn(&self) -> Result<String> {
        let value = run_llm_function(
            self.name().to_string(),
            vec!["_instructions".into(), "{}".into()],
            self.variable_envs(),
        )?;
        match value {
            Some(v) => Ok(v),
            _ => bail!("No return value from '_instructions' function"),
        }
    }

    /// `init_agent_variables`/`replace_tools_placeholder` were free functions/methods
    /// on the old `AgentDefinition`/`Agent` types; ported here unchanged in logic
    /// except `no_interaction: bool` -> `VariableInitMode` (see §3's two required
    /// behavior changes).
    pub fn init_agent_variables(
        agent_variables: &[AgentVariable],
        variables: &AgentVariables,
        mode: VariableInitMode,
    ) -> Result<AgentVariables> {
        let mut output = IndexMap::new();
        if agent_variables.is_empty() {
            return Ok(output);
        }
        let mut printed = false;
        let mut unset_variables = vec![];
        for agent_variable in agent_variables {
            let key = agent_variable.name.clone();
            match variables.get(&key) {
                Some(value) => {
                    output.insert(key, value.clone());
                }
                None => {
                    if let Some(value) = agent_variable.default.clone() {
                        output.insert(key, value);
                        continue;
                    }
                    match mode {
                        VariableInitMode::DefaultEmpty => {
                            output.insert(key, String::new());
                        }
                        VariableInitMode::FailOnMissing => {
                            unset_variables.push(agent_variable);
                        }
                        VariableInitMode::Interactive => {
                            if !printed {
                                println!("⚙ Init agent variables...");
                                printed = true;
                            }
                            let value = Text::new(&format!(
                                "{} ({}):",
                                agent_variable.name, agent_variable.description
                            ))
                            .with_validator(|input: &str| {
                                if input.trim().is_empty() {
                                    Ok(Validation::Invalid("This field is required".into()))
                                } else {
                                    Ok(Validation::Valid)
                                }
                            })
                            .prompt()?;
                            output.insert(key, value);
                        }
                    }
                }
            }
        }
        if !unset_variables.is_empty() {
            bail!(
                "The following agent variables are required:\n{}",
                unset_variables
                    .iter()
                    .map(|v| format!("  - {}: {}", v.name, v.description))
                    .collect::<Vec<_>>()
                    .join("\n")
            )
        }
        Ok(output)
    }

    fn replace_tools_placeholder(&mut self) {
        let tools_placeholder: &str = "{{__tools__}}";
        if self.prompt.contains(tools_placeholder) {
            let tools = self
                .functions
                .declarations()
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let description = match v.description.split_once('\n') {
                        Some((v, _)) => v,
                        None => &v.description,
                    };
                    format!("{}. {}: {description}", i + 1, v.name)
                })
                .collect::<Vec<String>>()
                .join("\n");
            self.prompt = self.prompt.replace(tools_placeholder, &tools);
        }
    }
}

impl RoleLike for Role {
    /// Almost no branching needed here compared to the old `Agent` impl, since
    /// `model`/`temperature`/`top_p`/`use_tools`/`use_agents`/`max_spawn_depth`/
    /// `max_subagent_turns` (and their setters) all read/write `self.settings`/
    /// `self.model` regardless of `kind`.
    fn to_role(&self) -> Role {
        let mut role = self.clone();
        role.kind = RoleKind::Role;
        role.prompt = self.effective_prompt();
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
        if !self.model().id().is_empty() {
            self.settings.model_id = Some(model.id().to_string());
        }
        self.model = model;
    }

    fn set_temperature(&mut self, value: Option<f64>) {
        self.settings.temperature = value;
    }

    fn set_top_p(&mut self, value: Option<f64>) {
        self.settings.top_p = value;
    }

    fn set_use_tools(&mut self, value: Option<String>) {
        self.settings.use_tools = value;
    }

    fn set_use_agents(&mut self, value: Option<String>) {
        self.settings.use_agents = value;
    }

    fn set_max_spawn_depth(&mut self, value: Option<usize>) {
        self.settings.max_spawn_depth = value;
    }

    fn set_max_subagent_turns(&mut self, value: Option<usize>) {
        self.settings.max_subagent_turns = value;
    }

    fn set_rag(&mut self, value: Option<Arc<Rag>>) {
        self.rag = value;
    }
}

fn parse_structure_prompt(prompt: &str) -> (&str, Vec<(&str, &str)>) {
    let mut text = prompt;
    let mut search_input = true;
    let mut system = None;
    let mut parts = vec![];
    loop {
        let search = if search_input {
            "### INPUT:"
        } else {
            "### OUTPUT:"
        };
        match text.find(search) {
            Some(idx) => {
                if system.is_none() {
                    system = Some(&text[..idx])
                } else {
                    parts.push(&text[..idx])
                }
                search_input = !search_input;
                text = &text[(idx + search.len())..];
            }
            None => {
                if !text.is_empty() {
                    if system.is_none() {
                        system = Some(text)
                    } else {
                        parts.push(text)
                    }
                }
                break;
            }
        }
    }
    let parts_len = parts.len();
    if parts_len > 0 && parts_len % 2 == 0 {
        let cases: Vec<(&str, &str)> = parts
            .iter()
            .step_by(2)
            .zip(parts.iter().skip(1).step_by(2))
            .map(|(i, o)| (i.trim(), o.trim()))
            .collect();
        let system = system.map(|v| v.trim()).unwrap_or_default();
        return (system, cases);
    }

    (prompt, vec![])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_structure_prompt1() {
        let prompt = r#"
System message
### INPUT:
Input 1
### OUTPUT:
Output 1
"#;
        assert_eq!(
            parse_structure_prompt(prompt),
            ("System message", vec![("Input 1", "Output 1")])
        );
    }

    #[test]
    fn test_parse_structure_prompt2() {
        let prompt = r#"
### INPUT:
Input 1
### OUTPUT:
Output 1
"#;
        assert_eq!(
            parse_structure_prompt(prompt),
            ("", vec![("Input 1", "Output 1")])
        );
    }

    #[test]
    fn test_parse_structure_prompt3() {
        let prompt = r#"
System message
### INPUT:
Input 1
"#;
        assert_eq!(parse_structure_prompt(prompt), (prompt, vec![]));
    }
}

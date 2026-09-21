use super::*;

use crate::client::{Message, MessageContent, MessageRole, Model};
use crate::function::{run_llm_function, Functions};

use anyhow::{Context, Result};
use inquire::{validator::Validation, Confirm, Text};
use rust_embed::Embed;
use serde::{Deserialize, Serialize};
use serde_json::Value;
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
        role.functions = Functions::init(&functions_path, &dir, true)?;
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

    /// The prompt template's own trailing, unmatched `### INPUT:` section --
    /// see `parse_structure_prompt` -- if it has one and it's non-empty
    /// after trimming. `None` for an empty prompt or an `__INPUT__`-embedded
    /// one (neither has any `### INPUT:`/`### OUTPUT:` structure at all, so
    /// the concept doesn't apply), or a structured prompt with no dangling
    /// section.
    ///
    /// This is the precise "does this role/agent have a pending
    /// pregeneration turn" check -- deliberately *not* "does
    /// `build_messages` end in `MessageRole::User`", which is true for
    /// almost every normal call regardless of whether there's anything
    /// genuinely pending (an ordinary empty-prompt role with empty input
    /// still produces `[User("")]`, a plain system prompt with no dangling
    /// section still ends up `[System(..), User("")]`, and so on) -- that
    /// was the actual bug in an earlier version of this check. See
    /// `crate::repl::maybe_complete_pending_turn`.
    pub fn dangling_input(&self) -> Option<String> {
        if self.is_empty_prompt() || self.is_embedded_prompt() {
            return None;
        }
        let prompt = self.effective_prompt();
        let (_, _, dangling) = parse_structure_prompt(&prompt);
        dangling.filter(|v| !v.is_empty()).map(|v| v.to_string())
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

    /// The last message in the returned list is `MessageRole::User` in every
    /// case *except* when `input.continue_output()` is set (a partial
    /// assistant reply is appended last instead, for `.continue`). In
    /// particular, when the prompt template has a trailing, unmatched
    /// `### INPUT:` section and `input` itself carries no real content (the
    /// common case: called right after activation, before any human input
    /// exists), that dangling section becomes the final message directly --
    /// no redundant empty turn on top of it. See `parse_structure_prompt`.
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
            let (system, cases, dangling_input) = parse_structure_prompt(&prompt);
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
            if let Some(dangling) = dangling_input {
                // Not a few-shot case (no scripted reply to pair with) -- a
                // turn the model is meant to complete. If there's no real
                // input to append after it, it simply *is* the final turn.
                // If there is real input too (unusual, but not disallowed),
                // it's inserted just before it rather than dropped -- see
                // this method's own doc comment.
                messages.push(Message::new(
                    MessageRole::User,
                    MessageContent::Text(dangling.to_string()),
                ));
            }
            if !input.is_empty() || dangling_input.is_none() {
                messages.push(Message::new(MessageRole::User, content));
            }
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

    pub fn update_shared_dynamic_instructions(
        &mut self,
        force: bool,
        input: Option<&str>,
        lua_tools_enabled: bool,
    ) -> Result<()> {
        if self.is_dynamic_instructions() && (force || self.shared_dynamic_instructions.is_none()) {
            self.shared_dynamic_instructions =
                Some(self.run_instructions_fn(input, lua_tools_enabled)?);
        }
        Ok(())
    }

    pub fn update_session_dynamic_instructions(
        &mut self,
        value: Option<String>,
        lua_tools_enabled: bool,
    ) -> Result<()> {
        if self.is_dynamic_instructions() {
            let value = match value {
                Some(v) => v,
                // Never called for a spawned subagent -- a subagent has no
                // session at all (`eval_spawn_subagent` calls
                // `update_shared_dynamic_instructions` instead, with its own
                // `input`), so there's no per-call `input` to thread through
                // here: always `None`.
                None => self.run_instructions_fn(None, lua_tools_enabled)?,
            };
            self.session_dynamic_instructions = Some(value);
        }
        Ok(())
    }

    /// Computes this agent's dynamic instructions text by invoking
    /// `_instructions`, checked in the same order a regular tool call would
    /// use (see `crate::lua_tool`'s module docs): a Lua `_instructions`
    /// function or `register()` entry directly inside this agent's own
    /// directory first (when `lua_tools_enabled`), falling back to the
    /// external executable at `<agent_dir>/bin/_instructions` -- same
    /// `bin/`-reserved-for-external-tools convention a regular tool call
    /// uses. Unlike a regular tool, `_instructions` is never offered to the
    /// model as a callable declaration -- it's invoked directly, by this
    /// reserved name -- so it doesn't need to be documented
    /// (`docs._instructions`/a comment block) to be found the way
    /// `crate::lua_tool::discover_declarations` would require of a
    /// model-facing tool.
    ///
    /// Runs the Lua VM inline via `crate::lua_tool::run_lua_tool_sync` rather
    /// than the `spawn_blocking`-wrapped `run_lua_tool` a regular tool call
    /// uses -- same trade-off `Functions::init`/`discover_declarations`
    /// already make and for the same reason: this only runs once at
    /// agent/session activation, not on the per-turn tool-call hot path
    /// `run_lua_tool` exists for, so it isn't worth threading async/lock-
    /// across-await concerns through the whole activation call chain for.
    ///
    /// `input` is the spawning `spawn_subagent` call's own `input` argument
    /// when this agent is being activated as a subagent, or `None`
    /// (serialized as JSON `null`) for every other activation path -- a
    /// plain `.agent`/`--agent` activation has no such input at all.
    fn run_instructions_fn(&self, input: Option<&str>, lua_tools_enabled: bool) -> Result<String> {
        let args = json!({ "input": input });

        if lua_tools_enabled {
            let dir = Config::agent_dir(self.name());
            if crate::lua_tool::has_lua_tools(&dir) {
                if let Some(output) = crate::lua_tool::run_lua_tool_sync(
                    &dir,
                    "_instructions",
                    args.clone(),
                    self.variable_envs(),
                )? {
                    return match output {
                        Value::String(s) => Ok(s),
                        other => {
                            bail!("'_instructions' must return a string, got: {other}")
                        }
                    };
                }
                // No Lua `_instructions` found in that directory -- fall
                // through to the external executable below, exactly as a
                // regular tool call would.
            }
        }

        let value = run_llm_function(
            self.name().to_string(),
            vec!["_instructions".into(), args.to_string()],
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

/// Parses a prompt template into its leading system text, any complete
/// `### INPUT:`/`### OUTPUT:` few-shot pairs, and -- new -- a trailing,
/// unmatched `### INPUT:` section, if the prompt ends with one (returned as
/// the third element). A dangling `### INPUT:` isn't a few-shot example (it
/// has no scripted reply to pair with); it's a turn meant to be completed by
/// the model, which is exactly what `Role::build_messages` uses it for.
///
/// Previously, an odd number of markers (i.e. exactly this dangling case)
/// discarded all parsing and returned the whole raw prompt, markers and all,
/// as one opaque system message -- see this function's tests for the
/// difference.
fn parse_structure_prompt(prompt: &str) -> (&str, Vec<(&str, &str)>, Option<&str>) {
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
    if parts.is_empty() {
        // No `### INPUT:`/`### OUTPUT:` structure at all -- unchanged from
        // before: the whole prompt is the system text, verbatim (not
        // `system.map(|v| v.trim())`, deliberately, to match prior behavior).
        return (prompt, vec![], None);
    }
    // An odd count means the last part was pushed right after a `### INPUT:`
    // match with no following `### OUTPUT:` before the prompt ended -- i.e.
    // it's the dangling input, not part of a pair. Pop it before pairing up
    // everything else.
    let dangling_input = if parts.len() % 2 == 1 {
        parts.pop().map(|v| v.trim())
    } else {
        None
    };
    let cases: Vec<(&str, &str)> = parts
        .iter()
        .step_by(2)
        .zip(parts.iter().skip(1).step_by(2))
        .map(|(i, o)| (i.trim(), o.trim()))
        .collect();
    let system = system.map(|v| v.trim()).unwrap_or_default();
    (system, cases, dangling_input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dangling_input_none_for_plain_prompts() {
        // Empty prompt -- no structure of any kind.
        assert_eq!(Role::new("t", "").dangling_input(), None);
        // `__INPUT__`-embedded prompt -- different mechanism entirely.
        assert_eq!(
            Role::new("t", &format!("Echo: {INPUT_PLACEHOLDER}")).dangling_input(),
            None
        );
        // A plain system prompt with no `### INPUT:`/`### OUTPUT:` markers
        // at all -- this is the ordinary case for almost every role/agent,
        // and must NOT be mistaken for a pending turn (this was the actual
        // bug: an earlier check fired here on every single activation).
        assert_eq!(
            Role::new("t", "You are a helpful assistant.").dangling_input(),
            None
        );
        // Complete `### INPUT:`/`### OUTPUT:` pairs, but nothing dangling.
        assert_eq!(
            Role::new("t", "System.\n### INPUT:\nIn.\n### OUTPUT:\nOut.\n").dangling_input(),
            None
        );
        // A dangling section that's empty/whitespace-only after trimming.
        assert_eq!(
            Role::new("t", "System.\n### INPUT:\n   \n").dangling_input(),
            None
        );
    }

    #[test]
    fn dangling_input_some_for_a_genuine_trailing_section() {
        assert_eq!(
            Role::new("t", "System.\n### INPUT:\nGenerated scene.\n").dangling_input(),
            Some("Generated scene.".to_string())
        );
    }

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
            ("System message", vec![("Input 1", "Output 1")], None)
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
            ("", vec![("Input 1", "Output 1")], None)
        );
    }

    #[test]
    fn test_parse_structure_prompt3() {
        // A trailing, unmatched `### INPUT:` -- no longer discards all
        // parsing (see this function's doc comment); it's returned as the
        // third element instead.
        let prompt = r#"
System message
### INPUT:
Input 1
"#;
        assert_eq!(
            parse_structure_prompt(prompt),
            ("System message", vec![], Some("Input 1"))
        );
    }

    #[test]
    fn test_parse_structure_prompt4() {
        // Complete few-shot pairs *and* a trailing dangling input together --
        // the `adventure`-agent shape (a static scene-setting example plus a
        // final turn meant to be completed).
        let prompt = r#"
System message
### INPUT:
Input 1
### OUTPUT:
Output 1
### INPUT:
Input 2
"#;
        assert_eq!(
            parse_structure_prompt(prompt),
            (
                "System message",
                vec![("Input 1", "Output 1")],
                Some("Input 2")
            )
        );
    }

    #[test]
    fn test_parse_structure_prompt_no_markers_at_all() {
        // No `### INPUT:`/`### OUTPUT:` structure whatsoever -- unchanged:
        // the whole prompt, untrimmed, becomes the system text.
        let prompt = "Just a plain system prompt, no structure.";
        assert_eq!(parse_structure_prompt(prompt), (prompt, vec![], None));
    }

    #[test]
    fn build_messages_uses_dangling_input_as_final_turn_when_input_is_empty() {
        let config: GlobalConfig = Arc::new(RwLock::new(Config::default()));
        let role = Role::new(
            "test",
            "System text.\n### INPUT:\nGenerated opening scene.\n",
        );
        let input = Input::from_str(&config, "", Some(role.clone()));
        let messages = role.build_messages(&input);
        // No redundant empty turn on top -- exactly system + the dangling
        // input as the final (User) message.
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, MessageRole::System);
        assert_eq!(messages[1].role, MessageRole::User);
        assert!(
            matches!(&messages[1].content, MessageContent::Text(t) if t == "Generated opening scene.")
        );
    }

    #[test]
    fn build_messages_keeps_both_turns_when_dangling_input_and_real_input_coexist() {
        let config: GlobalConfig = Arc::new(RwLock::new(Config::default()));
        let role = Role::new("test", "### INPUT:\nScene.\n");
        let input = Input::from_str(&config, "hello", Some(role.clone()));
        let messages = role.build_messages(&input);
        // Real input isn't dropped just because the prompt also has a
        // dangling section -- it's appended after it (see this method's doc
        // comment for the resulting, unusual-but-tolerated shape).
        assert_eq!(messages.len(), 2);
        assert!(messages.iter().all(|m| m.role == MessageRole::User));
        assert!(matches!(&messages[0].content, MessageContent::Text(t) if t == "Scene."));
        assert!(matches!(&messages[1].content, MessageContent::Text(t) if t == "hello"));
    }

    #[test]
    fn run_instructions_fn_prefers_lua_over_external() {
        let agent_name = format!(
            "aichat_test_instructions_agent_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let dir = std::env::temp_dir().join(&agent_name);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("_instructions.lua"),
            r#"
                function _instructions(input)
                    return "Lua instructions, input=" .. tostring(input)
                end
            "#,
        )
        .unwrap();
        let env_var = format!("{}_AGENT_DIR", normalize_env_name(&agent_name));
        std::env::set_var(&env_var, &dir);

        let role = Role {
            kind: RoleKind::Agent,
            name: agent_name.clone(),
            dynamic_instructions: true,
            ..Default::default()
        };

        let with_input = role.run_instructions_fn(Some("do the task"), true).unwrap();
        assert_eq!(with_input, "Lua instructions, input=do the task");

        let without_input = role.run_instructions_fn(None, true).unwrap();
        assert_eq!(without_input, "Lua instructions, input=nil");

        std::env::remove_var(&env_var);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn run_instructions_fn_requires_a_string_return() {
        let agent_name = format!(
            "aichat_test_instructions_nonstring_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let dir = std::env::temp_dir().join(&agent_name);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("_instructions.lua"),
            "function _instructions(input) return { oops = true } end",
        )
        .unwrap();
        let env_var = format!("{}_AGENT_DIR", normalize_env_name(&agent_name));
        std::env::set_var(&env_var, &dir);

        let role = Role {
            kind: RoleKind::Agent,
            name: agent_name.clone(),
            dynamic_instructions: true,
            ..Default::default()
        };

        let err = role.run_instructions_fn(None, true).unwrap_err();
        assert!(err.to_string().contains("must return a string"));

        std::env::remove_var(&env_var);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn run_instructions_fn_skips_lua_when_disabled() {
        let agent_name = format!(
            "aichat_test_instructions_disabled_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let dir = std::env::temp_dir().join(&agent_name);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("_instructions.lua"),
            "function _instructions() return \"should not be used\" end",
        )
        .unwrap();
        let env_var = format!("{}_AGENT_DIR", normalize_env_name(&agent_name));
        std::env::set_var(&env_var, &dir);

        let role = Role {
            kind: RoleKind::Agent,
            name: agent_name.clone(),
            dynamic_instructions: true,
            ..Default::default()
        };

        // `lua_tools_enabled: false` -- falls straight to the external-process
        // path, which fails here since there's no `bin/_instructions`
        // executable; the point is it does NOT return the Lua function's text.
        let err = role.run_instructions_fn(None, false).unwrap_err();
        assert!(!err.to_string().contains("should not be used"));

        std::env::remove_var(&env_var);
        let _ = std::fs::remove_dir_all(&dir);
    }
}

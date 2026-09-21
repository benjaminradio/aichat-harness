use crate::{
    config::{Config, GlobalConfig, Input, Role, RoleLike},
    utils::*,
};

use anyhow::{anyhow, bail, Context, Result};
use indexmap::IndexMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::{HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};

#[cfg(windows)]
const PATH_SEP: &str = ";";
#[cfg(not(windows))]
const PATH_SEP: &str = ":";

pub async fn eval_tool_calls(
    config: &GlobalConfig,
    mut calls: Vec<ToolCall>,
    abort_signal: AbortSignal,
) -> Result<Vec<ToolResult>> {
    let mut output = vec![];
    if calls.is_empty() {
        return Ok(output);
    }
    calls = ToolCall::dedup(calls);
    if calls.is_empty() {
        bail!("The request was aborted because an infinite loop of function calls was detected.")
    }
    let mut is_all_null = true;
    for call in calls {
        let (mut result, trace) = call.eval(config, abort_signal.clone()).await?;
        if result.is_null() {
            result = json!("DONE");
        } else {
            is_all_null = false;
        }
        output.push(match trace {
            Some(trace) => ToolResult::new_with_trace(call, result, trace),
            None => ToolResult::new(call, result),
        });
    }
    if is_all_null {
        output = vec![];
    }
    Ok(output)
}

#[derive(Debug, Clone)]
pub struct ToolResult {
    pub call: ToolCall,
    pub output: Value,
    /// Rich supplementary detail that should be persisted and displayed but never
    /// sent back to the model. Only `spawn_subagent` populates this today (the
    /// subagent's full internal message trace); everything else leaves it `None`.
    pub trace: Option<Value>,
}

impl ToolResult {
    pub fn new(call: ToolCall, output: Value) -> Self {
        Self {
            call,
            output,
            trace: None,
        }
    }

    pub fn new_with_trace(call: ToolCall, output: Value, trace: Value) -> Self {
        Self {
            call,
            output,
            trace: Some(trace),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Functions {
    declarations: Vec<FunctionDeclaration>,
}

impl Functions {
    /// Loads declarations from `declarations_path`'s `functions.json` (if it
    /// exists) and merges in any declarations discovered from the `.lua`
    /// files directly inside `lua_dir` (NOT a `bin/` subdirectory of it --
    /// that stays reserved for external-process tools; see
    /// `crate::lua_tool`'s module docs) via
    /// [`crate::lua_tool::discover_declarations`], whose name isn't already
    /// covered by `functions.json` -- an explicit `functions.json` entry
    /// always wins over discovery, so a hand-written declaration is still a
    /// way to override or annotate a Lua tool. `is_agent` sets the
    /// discovered declarations' `agent` flag, mirroring what an agent's own
    /// `functions.json` would set by hand.
    pub fn init(declarations_path: &Path, lua_dir: &Path, is_agent: bool) -> Result<Self> {
        let mut declarations: Vec<FunctionDeclaration> = if declarations_path.exists() {
            let ctx = || {
                format!(
                    "Failed to load functions at {}",
                    declarations_path.display()
                )
            };
            let content = fs::read_to_string(declarations_path).with_context(ctx)?;
            serde_json::from_str(&content).with_context(ctx)?
        } else {
            vec![]
        };

        if crate::lua_tool::has_lua_tools(lua_dir) {
            let existing: HashSet<String> = declarations.iter().map(|v| v.name.clone()).collect();
            let discovered = crate::lua_tool::discover_declarations(lua_dir, is_agent)
                .with_context(|| {
                    format!("Failed to discover Lua tools in {}", lua_dir.display())
                })?;
            for declaration in discovered {
                if !existing.contains(&declaration.name) {
                    declarations.push(declaration);
                }
            }
        }

        Ok(Self { declarations })
    }

    pub fn find(&self, name: &str) -> Option<&FunctionDeclaration> {
        self.declarations.iter().find(|v| v.name == name)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.declarations.iter().any(|v| v.name == name)
    }

    pub fn declarations(&self) -> &[FunctionDeclaration] {
        &self.declarations
    }

    pub fn is_empty(&self) -> bool {
        self.declarations.is_empty()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDeclaration {
    pub name: String,
    pub description: String,
    pub parameters: JsonSchema,
    #[serde(skip_serializing, default)]
    pub agent: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct JsonSchema {
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub type_value: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub properties: Option<IndexMap<String, JsonSchema>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub items: Option<Box<JsonSchema>>,
    #[serde(rename = "anyOf", skip_serializing_if = "Option::is_none")]
    pub any_of: Option<Vec<JsonSchema>>,
    #[serde(rename = "enum", skip_serializing_if = "Option::is_none")]
    pub enum_value: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required: Option<Vec<String>>,
}

impl JsonSchema {
    pub fn is_empty_properties(&self) -> bool {
        match &self.properties {
            Some(v) => v.is_empty(),
            None => true,
        }
    }
}

pub const SPAWN_SUBAGENT_TOOL_NAME: &str = "spawn_subagent";

/// The synthetic `spawn_subagent` declaration injected in `select_functions` between
/// tier 1 (global tools) and tier 2 (agent-declared tools), so that an agent
/// declaring its own `spawn_subagent` in `functions.json` shadows this one through
/// the existing name-dedup, with no special-casing.
pub fn spawn_subagent_declaration(allowed_agents: &[String]) -> FunctionDeclaration {
    let agent_schema = JsonSchema {
        type_value: Some("string".into()),
        description: Some("Name of the agent to run.".into()),
        // When the allowlist is a concrete set of names (rather than "all"), surface
        // it as an enum so the model can't invent an agent name that would just fail
        // the allowlist check at dispatch time.
        enum_value: if allowed_agents.is_empty() {
            None
        } else {
            Some(allowed_agents.to_vec())
        },
        ..Default::default()
    };
    let input_schema = JsonSchema {
        type_value: Some("string".into()),
        description: Some(
            "The task to give the subagent. It does not see this conversation, so state the task completely and self-containedly."
                .into(),
        ),
        ..Default::default()
    };
    let mut properties = IndexMap::new();
    properties.insert("agent".to_string(), agent_schema);
    properties.insert("input".to_string(), input_schema);
    FunctionDeclaration {
        name: SPAWN_SUBAGENT_TOOL_NAME.into(),
        description: "Run a task in an isolated subagent and return its final answer. The subagent starts with no history and cannot see this conversation.".into(),
        parameters: JsonSchema {
            type_value: Some("object".into()),
            properties: Some(properties),
            required: Some(vec!["agent".into(), "input".into()]),
            ..Default::default()
        },
        agent: false,
    }
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Value,
    pub id: Option<String>,
}

type CallConfig = (String, String, Vec<String>, HashMap<String, String>);

impl ToolCall {
    pub fn dedup(calls: Vec<Self>) -> Vec<Self> {
        let mut new_calls = vec![];
        let mut seen_ids = HashSet::new();

        for call in calls.into_iter().rev() {
            if let Some(id) = &call.id {
                if !seen_ids.contains(id) {
                    seen_ids.insert(id.clone());
                    new_calls.push(call);
                }
            } else {
                new_calls.push(call);
            }
        }

        new_calls.reverse();
        new_calls
    }

    pub fn new(name: String, arguments: Value, id: Option<String>) -> Self {
        Self {
            name,
            arguments,
            id,
        }
    }

    pub async fn eval(
        &self,
        config: &GlobalConfig,
        abort_signal: AbortSignal,
    ) -> Result<(Value, Option<Value>)> {
        // Interception point for tool kinds that don't go through
        // `run_llm_function`'s executable search at all:
        // - `spawn_subagent` is synthetic (it has no declaration in
        //   `Functions` at all), so it has to be intercepted here, before the
        //   `extract_call_config_from_agent`/`extract_call_config_from_config`
        //   resolution below even runs.
        // Lua-backed tools are NOT like that -- a Lua tool has a declaration
        // in `Functions` (either hand-written in `functions.json`, or
        // auto-discovered from its own Lua source -- see
        // `Functions::init`/`crate::lua_tool::discover_declarations`) and
        // goes through the exact same `cmd_name`/`cmd_args`/`envs`
        // resolution as an external-process tool. Only the final dispatch
        // step differs; see the `lua_tools` branch below, right before the
        // `run_llm_function` call it's a sibling to.
        if self.name == SPAWN_SUBAGENT_TOOL_NAME && !agent_declares_override(config, &self.name) {
            return self.eval_spawn_subagent(config, abort_signal).await;
        }

        let (call_name, cmd_name, mut cmd_args, envs) = match &config.read().agent {
            Some(agent) => self.extract_call_config_from_agent(config, agent)?,
            None => self.extract_call_config_from_config(config)?,
        };

        let json_data = if self.arguments.is_object() {
            self.arguments.clone()
        } else if let Some(arguments) = self.arguments.as_str() {
            let arguments: Value = serde_json::from_str(arguments).map_err(|_| {
                anyhow!("The call '{call_name}' has invalid arguments: {arguments}")
            })?;
            arguments
        } else {
            bail!(
                "The call '{call_name}' has invalid arguments: {}",
                self.arguments
            );
        };

        if config.read().lua_tools {
            let lua_dir = crate::lua_tool::lua_tools_dir_for(&cmd_name, &cmd_args);
            if crate::lua_tool::has_lua_tools(&lua_dir) {
                if let Some(output) = crate::lua_tool::run_lua_tool(
                    lua_dir,
                    self.name.clone(),
                    json_data.clone(),
                    envs.clone(),
                )
                .await?
                {
                    return Ok((output, None));
                }
                // No tool named `self.name` was found among the `.lua` files in
                // that directory -- fall through to the executable search below,
                // exactly as if this feature didn't exist for this call.
            }
        }

        cmd_args.push(json_data.to_string());

        let output = match run_llm_function(cmd_name, cmd_args, envs)? {
            Some(contents) => serde_json::from_str(&contents)
                .ok()
                .unwrap_or_else(|| json!({"output": contents})),
            None => Value::Null,
        };

        Ok((output, None))
    }

    fn extract_call_config_from_agent(
        &self,
        config: &GlobalConfig,
        agent: &Role,
    ) -> Result<CallConfig> {
        let function_name = self.name.clone();
        match agent.functions().find(&function_name) {
            Some(function) => {
                let agent_name = agent.name().to_string();
                if function.agent {
                    Ok((
                        format!("{agent_name}-{function_name}"),
                        agent_name,
                        vec![function_name],
                        agent.variable_envs(),
                    ))
                } else {
                    Ok((
                        function_name.clone(),
                        function_name,
                        vec![],
                        Default::default(),
                    ))
                }
            }
            None => self.extract_call_config_from_config(config),
        }
    }

    fn extract_call_config_from_config(&self, config: &GlobalConfig) -> Result<CallConfig> {
        let function_name = self.name.clone();
        match config.read().functions.contains(&function_name) {
            true => Ok((
                function_name.clone(),
                function_name,
                vec![],
                Default::default(),
            )),
            false => bail!("Unexpected call: {function_name} {}", self.arguments),
        }
    }

    /// Run a subagent in isolation and return its final assistant text.
    ///
    /// Follows `Config::macro_execute`'s existing isolation pattern (clone the whole
    /// `Config`, reset context fields, drive it under a fresh `Arc<RwLock<Config>>`)
    /// rather than inventing a new mechanism. Nothing is written back to the parent
    /// config or its session — the child handle is simply dropped at the end.
    ///
    /// `#[async_recursion]` is required, not stylistic: this sits in a genuine async
    /// cycle (eval → eval_spawn_subagent → run_completion_loop → call_chat_completions
    /// → eval_tool_calls → eval), which would otherwise be an infinitely-sized future.
    /// Boxing here is what breaks it.
    #[async_recursion::async_recursion]
    async fn eval_spawn_subagent(
        &self,
        config: &GlobalConfig,
        abort_signal: AbortSignal,
    ) -> Result<(Value, Option<Value>)> {
        let args = if self.arguments.is_object() {
            self.arguments.clone()
        } else if let Some(arguments) = self.arguments.as_str() {
            serde_json::from_str(arguments)
                .map_err(|_| anyhow!("The call 'spawn_subagent' has invalid arguments"))?
        } else {
            bail!("The call 'spawn_subagent' has invalid arguments");
        };
        let agent_name = args
            .get("agent")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("spawn_subagent requires an 'agent' argument"))?
            .to_string();
        let input_text = args
            .get("input")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("spawn_subagent requires an 'input' argument"))?
            .to_string();

        let (parent_role, parent_depth, parent_ceiling, stream) = {
            let cfg = config.read();
            (
                cfg.extract_role(),
                cfg.agent_depth,
                cfg.agent_depth_ceiling,
                cfg.stream,
            )
        };

        // Re-check the allowlist at dispatch time. The synthetic declaration already
        // constrains `agent` to an enum of allowed names, but a model can still emit
        // anything, and an agent-declared override reaches a different path entirely.
        let allowed = config.read().allowed_agents(&parent_role);
        if !allowed.contains(&agent_name) {
            bail!("Agent '{agent_name}' is not in the allowlist for spawning subagents");
        }
        // Defense in depth: select_functions only offers the tool below the ceiling,
        // but a model may replay a stale call from earlier context.
        if parent_depth >= config.read().effective_max_spawn_depth(&parent_role) {
            bail!("Maximum subagent spawn depth reached");
        }

        let mut child = config.read().clone();
        child.role = None;
        child.session = None;
        child.agent = None;
        child.harness_active = false;
        child.harness_activated_agent = false;
        child.agent_depth = parent_depth + 1;
        child.agent_depth_ceiling = if parent_ceiling == 0 {
            usize::MAX
        } else {
            parent_ceiling
        };
        let child_config: GlobalConfig = Arc::new(RwLock::new(child));

        // force_init_rag: true — a subagent has no interactive terminal to answer the
        // "init RAG?" confirmation, so build the index rather than silently running
        // without the documents the agent expects.
        let agent = Role::load_agent(&child_config, &agent_name, true, abort_signal.clone()).await?;
        // A subagent may tighten the ceiling for its own descendants, never raise it.
        {
            let mut cfg = child_config.write();
            if let Some(own) = agent.max_spawn_depth() {
                cfg.agent_depth_ceiling = cfg.agent_depth_ceiling.min(own);
            }
            cfg.agent = Some(agent);
        }
        // `Some(&input_text)` here is what lets a Lua/external `_instructions`
        // for this agent see the task it's being spawned to do -- `input_text`
        // is already in scope (extracted at the top of this function), so no
        // reordering is needed to thread it through.
        child_config
            .write()
            .init_agent_shared_variables(Some(&input_text))?;

        let max_turns = child_config
            .read()
            .agent
            .as_ref()
            .and_then(|v| v.max_subagent_turns());

        let input = Input::from_str(&child_config, &input_text, None);

        let show_subagent = config.read().show_subagent;

        // Sequential dispatch within eval_tool_calls guarantees no concurrent
        // subagent stream can interleave with this one, so plain open/close tags are
        // unambiguous. Mirrors how reasoning is wrapped in <think> tags. Dimmed
        // (rather than a bare `println!`) so they read as a boundary marker in the
        // terminal instead of flat, unstyled text -- everything else printed here
        // (tool-call traces, etc.) already goes through `dimmed_text` the same way.
        //
        // When `show_subagent` is off, these -- and the subagent's own nested
        // stream, suppressed independently in `render::render_stream`/
        // `harness::run_completion_loop` via `agent_depth` -- are replaced by a
        // "Subagent" spinner for the call's duration instead: the result and its
        // trace (for `.info session`) are unaffected either way, only what's shown.
        let mut spinner = None;
        if stream {
            if show_subagent {
                println!("{}", dimmed_text(&format!("<subagent agent=\"{agent_name}\">")));
            } else {
                spinner = Some(spawn_spinner("Subagent"));
            }
        }
        let result =
            crate::harness::run_completion_loop(&child_config, input, false, max_turns, abort_signal)
                .await;
        if let Some(spinner) = spinner.take() {
            spinner.stop();
        }
        if stream && show_subagent {
            println!("{}", dimmed_text("</subagent>"));
        }

        let outcome = result?;
        let trace = json!({
            "agent": agent_name,
            "messages": outcome.messages,
        });
        Ok((json!(outcome.output), Some(trace)))
    }
}

/// Whether the active agent declares its own `spawn_subagent` in `functions.json`.
/// If it does, `select_functions`' tier-2 dedup already shadowed the synthetic
/// declaration, so the model never saw ours and the call belongs to normal
/// external-process dispatch.
fn agent_declares_override(config: &GlobalConfig, name: &str) -> bool {
    match &config.read().agent {
        Some(agent) => agent
            .functions()
            .declarations()
            .iter()
            .any(|v| v.name == name),
        None => false,
    }
}

pub fn run_llm_function(
    cmd_name: String,
    cmd_args: Vec<String>,
    mut envs: HashMap<String, String>,
) -> Result<Option<String>> {
    let prompt = format!("Call {cmd_name} {}", cmd_args.join(" "));

    let mut bin_dirs: Vec<PathBuf> = vec![];
    if cmd_args.len() > 1 {
        let dir = Config::agent_dir(&cmd_name).join("bin");
        if dir.exists() {
            bin_dirs.push(dir);
        }
    }
    bin_dirs.push(Config::functions_bin_dir());
    let current_path = std::env::var("PATH").context("No PATH environment variable")?;
    let prepend_path = bin_dirs
        .iter()
        .map(|v| format!("{}{PATH_SEP}", v.display()))
        .collect::<Vec<_>>()
        .join("");
    envs.insert("PATH".into(), format!("{prepend_path}{current_path}"));

    let temp_file = temp_file("-eval-", "");
    envs.insert("LLM_OUTPUT".into(), temp_file.display().to_string());

    #[cfg(windows)]
    let cmd_name = polyfill_cmd_name(&cmd_name, &bin_dirs);
    if *IS_STDOUT_TERMINAL {
        println!("{}", dimmed_text(&prompt));
    }
    let exit_code = run_command(&cmd_name, &cmd_args, Some(envs))
        .map_err(|err| anyhow!("Unable to run {cmd_name}, {err}"))?;
    if exit_code != 0 {
        bail!("Tool call exit with {exit_code}");
    }
    let mut output = None;
    if temp_file.exists() {
        let contents =
            fs::read_to_string(temp_file).context("Failed to retrieve tool call output")?;
        if !contents.is_empty() {
            output = Some(contents);
        }
    };
    Ok(output)
}

#[cfg(windows)]
fn polyfill_cmd_name<T: AsRef<Path>>(cmd_name: &str, bin_dir: &[T]) -> String {
    let cmd_name = cmd_name.to_string();
    if let Ok(exts) = std::env::var("PATHEXT") {
        for name in exts.split(';').map(|ext| format!("{cmd_name}{ext}")) {
            for dir in bin_dir {
                let path = dir.as_ref().join(&name);
                if path.exists() {
                    return name.to_string();
                }
            }
        }
    }
    cmd_name
}

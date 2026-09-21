//! Lua-backed tool execution.
//!
//! A tool's *implementation* can live in a `.lua` file instead of (or as
//! well as) an external executable, alongside the process-based search in
//! [`crate::function::run_llm_function`]. Unlike an external-process tool --
//! which still must be hand-declared in `functions.json`, since there's no
//! way to introspect an arbitrary executable -- a Lua tool's declaration
//! (name, description, JSON-schema parameters) is derived directly from the
//! Lua source itself: `register()`'s own fields, `-- @param`/`docs.<name>`
//! documentation, and, as a last resort, the handler's actual parameter
//! names (via Lua's own introspection). `functions.json` still works if a
//! tool is declared there (explicit entries always win over discovery, see
//! [`crate::function::Functions::init`]), but for a Lua tool it's now
//! optional rather than required.
//!
//! ## Discovery
//!
//! Standalone tool -> every `*.lua` file directly inside
//! `Config::functions_dir()` (the same directory `functions.json` lives in --
//! not its `bin/` subdirectory, which stays reserved for external-process
//! tools; see [`crate::function::run_llm_function`]'s `bin_dirs`).
//! Agent-dispatcher tool -> every `*.lua` file directly inside
//! `Config::agent_dir(agent_name)` itself (again, not its `bin/`
//! subdirectory, for the same reason). Non-`.lua` entries in that directory
//! (`functions.json`, a `bin/` subdirectory, anything else) are simply
//! ignored by the scan -- no special-casing needed, the filter is purely by
//! extension.
//!
//! All `.lua` files in that one directory are loaded into a single shared Lua
//! VM, in filename order. This lets an author put several related tools --
//! or a tool plus private helper functions it calls -- in one file, or split
//! one logical group of tools across several files that share globals/
//! helpers, rather than requiring one file per tool.
//!
//! Two separate things read that directory: [`run_lua_tool`] (dispatch, one
//! specific call) and [`discover_declarations`] (building the
//! `functions.json`-equivalent declaration list, once per config/agent
//! load). Both spin up a fresh VM and reload every file -- no pooling, no
//! caching -- matching the mental model external-process tools already have
//! (a fresh process per invocation). Dispatch happens far more often than
//! discovery, so its cost matters more; if this directory contains no `.lua`
//! files at all, dispatch should skip straight past it (see
//! [`has_lua_tools`]), keeping the cost for anyone not using this feature to
//! a single cheap `read_dir`.
//!
//! ## Where this sits in the tool-call dispatch order
//!
//! `ToolCall::eval` (`function.rs`) tries, in order, for every tool call: (1)
//! the synthetic `spawn_subagent` tool, intercepted before any declaration
//! lookup at all, unless the active agent declares its own `spawn_subagent`
//! (in which case that declaration -- Lua or external -- is used instead,
//! via (2)/(3)); (2) a Lua tool, this module, via [`run_lua_tool`]; (3),
//! only if neither of those matched, the external-process search in
//! [`crate::function::run_llm_function`]. A directory with no `.lua` files
//! costs a single `read_dir` at step (2) before falling through to (3), so
//! nothing about this ordering has a cost for a tool that's purely external.
//!
//! ## Registering a tool from Lua
//!
//! Two ways, freely mixable within and across files in the same directory:
//!
//! 1. A **global function** named after the tool, documented via `docs.<name>`
//!    or via `---`/`-- @param` comments. Documentation is what makes a global
//!    function a *tool*: an undocumented global function is treated as a
//!    private helper and never exposed to the model (see
//!    [`discover_declarations`]).
//!
//!    ```lua
//!    docs.reverse_text = "Reverse the characters of a string."
//!    function reverse_text(text)
//!        return text:reverse()
//!    end
//!
//!    -- A tool that would shell out in real life (unrestricted `os`/`io` --
//!    -- see "Sandboxing" below):
//!    function get_weather(location, unit)
//!        unit = unit or "celsius"
//!        -- local handle = io.popen("curl -s 'https://wttr.in/" .. location .. "?format=3'")
//!        -- local result = handle:read("*a")
//!        -- handle:close()
//!        -- return result
//!        return "It is sunny and 22 degrees " .. unit .. " in " .. location .. "."
//!    end
//!    ```
//!
//! 2. An explicit **`register()`** call, which also allows several tools (or
//!    an options table) to live under one file without relying on Lua's
//!    global-name-equals-tool-name convention. `params` entries may be
//!    JSON-Schema-shaped (`type`, `enum`, `required`, `description`
//!    alongside `name`) -- these feed directly into the generated
//!    `functions.json`-equivalent declaration, not just documentation:
//!
//!    ```lua
//!    register("web_search", {
//!        description = "Search the web and return matching titles.",
//!        params = {
//!            { name = "query", type = "string", description = "Search query" },
//!            {
//!                name = "category",
//!                type = "string",
//!                description = "Restrict results to a category",
//!                enum = { "news", "images", "videos" },
//!                required = false,
//!            },
//!            { name = "limit", type = "integer", description = "Max results to return", required = false },
//!        },
//!        handler = function(query, category, limit)
//!            limit = limit or 3
//!            category = category or "general"
//!            return "Top " .. limit .. " " .. category .. " results for '" .. query .. "': ..."
//!        end,
//!    })
//!    ```
//!
//! A handler can **return any JSON-representable value**: a plain string (as
//! above) is the common case and is what a model reads most naturally as a
//! tool result -- a wrapped `{ result = "..." }` table just becomes visible
//! JSON-object noise in the reply for no benefit. Reach for a table only
//! when the result is genuinely structured (multiple named fields, a list of
//! results, etc.) -- see [`lua_to_json`] for exactly how Lua values map to
//! JSON.
//!
//! Both styles receive **positional** arguments matching the parameter names
//! (this is the "user friendly" part: no `args.foo` unpacking required), plus
//! a trailing `ctx` table carrying agent variables (`ctx.env.LLM_AGENT_VAR_*`,
//! same keys the `LLM_AGENT_VAR_*` env-var convention already uses for
//! external-process tools). Lua silently drops call arguments a function
//! didn't declare a parameter for, so a handler that doesn't care about `ctx`
//! can simply omit it from its parameter list.
//!
//! Parameter *names* (used both for the generated declaration's JSON-schema
//! `properties`, and to turn the JSON call arguments object into positional
//! call arguments at dispatch time -- always the same list, computed the same
//! way, so the two stay in sync by construction) are resolved per tool via
//! [`resolve_param_meta`], in this order:
//! 1. `register()`'s own `params` list, if given (an array of strings, or of
//!    `{ name = "...", ... }` tables).
//! 2. Otherwise, `-- @param name description` comment lines, if any, in the
//!    order written.
//! 3. Otherwise, the handler's own declared parameter names, read directly
//!    off the Lua function via `debug.getlocal` (see "Sandboxing" below) --
//!    so a plain `function reverse_text(text)` with no `params`/`@param`
//!    annotations at all still gets a correct, single-parameter-named-`text`
//!    declaration for free.
//!
//! Every param resolved this way defaults to JSON-schema `type: "string"`
//! and `required: true` unless a `register()` entry says otherwise; there's
//! no way to infer either from a bare comment or a plain function signature.
//!
//! Any JSON object field sent by the caller that isn't one of the resolved
//! parameter names is still passed through, positionally, after them, for a
//! variadic (`...`) handler to collect -- nothing the caller sent is
//! silently dropped even if the resolved parameter list is incomplete.
//!
//! ## Sandboxing
//!
//! Per an explicit, deliberate choice (matching today's external-process tool
//! trust model, where a tool is already an arbitrary executable): the Lua VM
//! ([`new_vm`]) loads *every* standard library, `StdLib::ALL` -- unrestricted
//! `os`/`io` (`os.execute`, `io.popen`, arbitrary file access, etc.), plus
//! `debug` (needed for the parameter-name introspection in
//! [`signature_param_names`], see above). Loading `debug` requires mlua's
//! `unsafe_new_with` rather than the default `Lua::new()` (which loads
//! `StdLib::ALL_SAFE`, everything *except* `debug`) -- not because `debug`
//! itself is a soundness hazard here, but because mlua reserves `unsafe` for
//! anything beyond its default safe subset. Sandboxing a Lua tool, if ever
//! wanted, is the tool author's responsibility (e.g. don't shell out in a
//! script you don't trust), not something this module imposes. See
//! `config.example.yaml`'s `lua_tools` setting for a way to disable this
//! backend entirely instead.
//!
//! ## Execution model
//!
//! Lua execution is synchronous and blocking. [`run_lua_tool`] (per-call
//! dispatch) hands the actual work to [`tokio::task::spawn_blocking`];
//! [`discover_declarations`] (config/agent-load-time only, not on the
//! per-call hot path) runs inline instead, same as the plain
//! `fs::read_to_string` + `serde_json::from_str` it sits alongside in
//! `Functions::init`. Either way, the `Lua` VM, its registry, and every
//! `mlua::Value`/`mlua::Function` created along the way are constructed
//! *and* consumed entirely within one synchronous call -- none of it is
//! moved across an `.await` point or a thread boundary in either direction
//! (only plain owned `String`/`serde_json::Value`/`HashMap`/
//! `FunctionDeclaration` data crosses those). That means this module does
//! not need mlua's `send` feature: `Lua`/`Function`/`Table` never need to be
//! `Send` here.

use crate::config::Config;
use crate::function::{FunctionDeclaration, JsonSchema};

use anyhow::{anyhow, bail, Context, Result};
use indexmap::IndexMap;
use serde_json::{json, Map, Value};
use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
    rc::Rc,
};

/// The directory this call's Lua tools (if any) would live in, mirroring
/// `run_llm_function`'s own `bin_dirs` logic but pointed at the *parent* of
/// its `bin/` (that stays reserved for external-process tools -- see the
/// module docs' "Discovery" section). `cmd_args` here is the `CallConfig`
/// shape *before* the JSON arguments string is pushed onto it (see
/// `function.rs`'s `eval`): agent-dispatch is exactly one element (the
/// function name, from `extract_call_config_from_agent`'s `function.agent`
/// branch), a plain standalone or non-agent-flagged call is empty.
pub fn lua_tools_dir_for(cmd_name: &str, cmd_args: &[String]) -> PathBuf {
    if !cmd_args.is_empty() {
        Config::agent_dir(cmd_name)
    } else {
        Config::functions_dir()
    }
}

/// Cheap existence check so a call with no Lua tools anywhere near it pays
/// only for a single `read_dir`, not for spinning up a Lua VM.
pub fn has_lua_tools(dir: &Path) -> bool {
    let Ok(entries) = fs::read_dir(dir) else {
        return false;
    };
    entries.filter_map(|e| e.ok()).any(|e| is_lua_file(&e.path()))
}

fn is_lua_file(path: &Path) -> bool {
    path.is_file() && path.extension().is_some_and(|ext| ext == "lua")
}

/// The Lua VM used for both dispatch and declaration discovery. Loads every
/// standard library (`StdLib::ALL`, including `debug` and unrestricted
/// `os`/`io`) -- see the module docs' "Sandboxing" section for why.
fn new_vm() -> mlua::Lua {
    // Safety: mlua's `unsafe` here is about exposing Lua's `debug` library to
    // the script being loaded, not about any unsound Rust operation in this
    // call itself -- see "Sandboxing" above for why that's an accepted,
    // deliberate part of this module's trust model.
    unsafe { mlua::Lua::unsafe_new_with(mlua::StdLib::ALL, mlua::LuaOptions::new()) }
}

/// Run the Lua tool named `tool_name` if one is found among the `.lua` files
/// in `dir`. Returns `Ok(None)` (not an error) if no such tool is found --
/// the caller should fall through to the existing external-process search in
/// that case, exactly as if this feature didn't exist for this call.
pub async fn run_lua_tool(
    dir: PathBuf,
    tool_name: String,
    args: Value,
    envs: HashMap<String, String>,
) -> Result<Option<Value>> {
    tokio::task::spawn_blocking(move || run_lua_tool_sync(&dir, &tool_name, args, envs))
        .await
        .context("The Lua tool task panicked")?
}

/// One resolved JSON-schema-ish parameter, however it was sourced (an
/// explicit `register()` `params` entry, a `-- @param` comment, or bare
/// signature introspection) -- see the module docs' "Parameter order"
/// section for the precedence between those sources. Only `name` is
/// guaranteed; everything else defaults sensibly (`type: "string"`,
/// `required: true`) when the source didn't say.
#[derive(Clone)]
struct ParamMeta {
    name: String,
    description: Option<String>,
    type_name: Option<String>,
    enum_values: Option<Vec<String>>,
    required: bool,
}

/// A tool registered via `register()`. Distinct from the plain
/// global-function case (which has no separate struct -- dispatch and
/// discovery just look the function up in `lua.globals()` directly) because
/// `register()` carries its own explicit `description`/`params` alongside
/// the handler.
#[derive(Clone)]
struct RegisteredTool {
    description: Option<String>,
    /// `None` means "no explicit `params` given" -- falls through to
    /// comment/signature resolution, same as a plain global function; see
    /// [`resolve_param_meta`].
    params: Option<Vec<ParamMeta>>,
    handler: mlua::Function,
}

/// Same dispatch as [`run_lua_tool`], but run inline rather than handed to
/// [`tokio::task::spawn_blocking`]. Exists for callers on a path that (a)
/// isn't the per-turn tool-call hot path [`run_lua_tool`] exists for and (b)
/// doesn't already have an async call chain worth extending just for this --
/// e.g. `Role::run_instructions_fn`, which runs once at agent/session
/// activation, same trade-off `Functions::init`/[`discover_declarations`]
/// already make for the same reason (see their own doc comments). Prefer
/// [`run_lua_tool`] for anything on the per-call dispatch path.
pub(crate) fn run_lua_tool_sync(
    dir: &Path,
    tool_name: &str,
    args: Value,
    envs: HashMap<String, String>,
) -> Result<Option<Value>> {
    let mut lua_files: Vec<PathBuf> = fs::read_dir(dir)
        .with_context(|| format!("Failed to read Lua tools directory {}", dir.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| is_lua_file(p))
        .collect();
    if lua_files.is_empty() {
        return Ok(None);
    }
    // Deterministic load order. Later files win on a global-function-name or
    // `register()`-name collision, same "last one wins, keep it simple" rule
    // the discovery convention already uses for `.lua` vs. same-named
    // executable.
    lua_files.sort();

    let lua = new_vm();
    let registry: Rc<RefCell<IndexMap<String, RegisteredTool>>> =
        Rc::new(RefCell::new(IndexMap::new()));
    let docs_table = lua
        .create_table()
        .map_err(|e| anyhow!("Failed to create the Lua `docs` table: {e}"))?;
    lua.globals()
        .set("docs", docs_table.clone())
        .map_err(|e| anyhow!("Failed to install the Lua `docs` global: {e}"))?;
    install_register_fn(&lua, registry.clone())
        .map_err(|e| anyhow!("Failed to install the Lua `register` global: {e}"))?;

    let mut doc_comments: HashMap<String, ParsedDocComment> = HashMap::new();
    for path in &lua_files {
        let source = fs::read_to_string(path)
            .with_context(|| format!("Failed to read Lua tool file {}", path.display()))?;
        doc_comments.extend(parse_doc_comments(&source));
        lua.load(source.as_str())
            .set_name(path.display().to_string())
            .exec()
            .map_err(|e| anyhow!("Failed to load Lua tool file {}: {e}", path.display()))?;
    }

    let (registered_handler, explicit_params, registered_description) = {
        let reg = registry.borrow();
        match reg.get(tool_name) {
            Some(tool) => (
                Some(tool.handler.clone()),
                tool.params.clone(),
                tool.description.clone(),
            ),
            None => (None, None, None),
        }
    };
    let handler = match registered_handler {
        Some(handler) => handler,
        None => {
            let value: mlua::Value = lua
                .globals()
                .get(tool_name)
                .map_err(|e| anyhow!("Failed to look up global Lua function '{tool_name}': {e}"))?;
            match value {
                mlua::Value::Function(handler) => handler,
                _ => return Ok(None),
            }
        }
    };

    let comment = doc_comments.get(tool_name);
    let description = registered_description
        .or_else(|| docs_table.get::<Option<String>>(tool_name).ok().flatten())
        .or_else(|| comment.and_then(|d| d.description.clone()));

    if *crate::utils::IS_STDOUT_TERMINAL {
        let suffix = description
            .as_deref()
            .map(|d| format!(" -- {d}"))
            .unwrap_or_default();
        println!(
            "{}",
            crate::utils::dimmed_text(&format!("Call {tool_name} (lua){suffix}"))
        );
    }

    let comment_params = comment.map(|c| c.params.as_slice());
    let params = resolve_param_meta(&lua, &handler, explicit_params.as_deref(), comment_params)
        .map_err(|e| anyhow!("Failed to inspect Lua tool '{tool_name}': {e}"))?;
    let param_names: Vec<String> = params.into_iter().map(|p| p.name).collect();

    let mut call_args = build_call_args(&lua, &args, &param_names)
        .map_err(|e| anyhow!("Failed to convert JSON arguments to Lua: {e}"))?;
    let ctx = build_ctx_table(&lua, &envs)
        .map_err(|e| anyhow!("Failed to build the Lua `ctx` table: {e}"))?;
    call_args.push(mlua::Value::Table(ctx));

    let result: mlua::Value = handler
        .call(mlua::Variadic::from(call_args))
        .map_err(|err| anyhow!("Lua tool '{tool_name}' raised an error: {err}"))?;

    Ok(Some(lua_to_json(result)?))
}

fn install_register_fn(
    lua: &mlua::Lua,
    registry: Rc<RefCell<IndexMap<String, RegisteredTool>>>,
) -> mlua::Result<()> {
    let register_fn = lua.create_function(move |_, (name, opts): (String, mlua::Table)| {
        let description: Option<String> = opts.get("description")?;
        let params = extract_params_field(&opts)?;
        let handler: mlua::Function = opts.get("handler").map_err(|_| {
            mlua::Error::RuntimeError(format!(
                "register(\"{name}\", ...) requires a `handler` function"
            ))
        })?;
        registry.borrow_mut().insert(
            name,
            RegisteredTool {
                description,
                params,
                handler,
            },
        );
        Ok(())
    })?;
    lua.globals().set("register", register_fn)
}

/// Parses `opts.params`, accepting either a flat array of strings
/// (`{ "query", "limit" }`) or an array of JSON-Schema-shaped tables
/// (`{ { name = "query", type = "string", description = "...", enum = {...},
/// required = false }, ... }` -- only `name` is required, everything else is
/// optional and defaults as described on [`ParamMeta`]).
fn extract_params_field(opts: &mlua::Table) -> mlua::Result<Option<Vec<ParamMeta>>> {
    let params_table: Option<mlua::Table> = opts.get("params")?;
    let Some(params_table) = params_table else {
        return Ok(None);
    };
    let len = params_table.raw_len();
    let mut params = Vec::with_capacity(len);
    for i in 1..=len as i64 {
        if let Ok(name) = params_table.get::<String>(i) {
            params.push(ParamMeta {
                name,
                description: None,
                type_name: None,
                enum_values: None,
                required: true,
            });
            continue;
        }
        let entry: mlua::Table = params_table.get(i).map_err(|_| {
            mlua::Error::RuntimeError(format!(
                "`params[{i}]` must be a string or a table with a `name` field"
            ))
        })?;
        let name: String = entry.get("name").map_err(|_| {
            mlua::Error::RuntimeError(format!("`params[{i}]` table is missing a `name` field"))
        })?;
        let description: Option<String> = entry.get("description")?;
        let type_name: Option<String> = entry.get("type")?;
        let enum_values: Option<Vec<String>> = entry.get("enum")?;
        let required: bool = entry.get::<Option<bool>>("required")?.unwrap_or(true);
        params.push(ParamMeta {
            name,
            description,
            type_name,
            enum_values,
            required,
        });
    }
    Ok(Some(params))
}

/// Resolves the final, ordered parameter list for one tool, per the module
/// docs' "Parameter order" precedence: explicit `params` (from `register()`)
/// first, then `-- @param` comment entries, then bare signature
/// introspection as the last resort. Shared by dispatch (which only needs
/// the names, to build positional call arguments) and
/// [`discover_declarations`] (which needs the full metadata, to build a
/// JSON-schema `properties` entry per parameter).
fn resolve_param_meta(
    lua: &mlua::Lua,
    handler: &mlua::Function,
    explicit_params: Option<&[ParamMeta]>,
    comment_params: Option<&[(String, String)]>,
) -> mlua::Result<Vec<ParamMeta>> {
    if let Some(params) = explicit_params {
        return Ok(params.to_vec());
    }
    if let Some(comment_params) = comment_params {
        if !comment_params.is_empty() {
            return Ok(comment_params
                .iter()
                .map(|(name, desc)| ParamMeta {
                    name: name.clone(),
                    description: (!desc.is_empty()).then(|| desc.clone()),
                    type_name: None,
                    enum_values: None,
                    required: true,
                })
                .collect());
        }
    }
    let names = signature_param_names(lua, handler)?;
    Ok(names
        .into_iter()
        .map(|name| ParamMeta {
            name,
            description: None,
            type_name: None,
            enum_values: None,
            required: true,
        })
        .collect())
}

/// Reads a Lua function's own declared parameter *names* (not values, and
/// not counting a trailing `...`) without calling it, via
/// `debug.getlocal(f, i)` -- documented in the Lua 5.4 manual as a variant of
/// `debug.getlocal` that accepts a function in place of a stack level for
/// exactly this purpose. `debug.getlocal` returns `nil` once `i` exceeds the
/// function's declared parameter count (not an error), which is what ends
/// the loop here.
fn signature_param_names(lua: &mlua::Lua, f: &mlua::Function) -> mlua::Result<Vec<String>> {
    let debug: mlua::Table = lua.globals().get("debug")?;
    let getlocal: mlua::Function = debug.get("getlocal")?;
    let mut names = Vec::new();
    let mut i: i64 = 1;
    loop {
        let name: Option<String> = getlocal.call((f.clone(), i))?;
        match name {
            Some(n) => names.push(n),
            None => break,
        }
        i += 1;
    }
    Ok(names)
}

fn build_ctx_table(lua: &mlua::Lua, envs: &HashMap<String, String>) -> mlua::Result<mlua::Table> {
    let ctx = lua.create_table()?;
    let env_table = lua.create_table()?;
    for (k, v) in envs {
        env_table.set(k.as_str(), v.as_str())?;
    }
    ctx.set("env", env_table)?;
    Ok(ctx)
}

/// Turns the JSON arguments object into a positional Lua argument list: named
/// parameters (in `param_order`) first, then any leftover JSON fields (in
/// their original order) for a variadic handler to pick up. The trailing
/// `ctx` table is appended by the caller, not here.
fn build_call_args(
    lua: &mlua::Lua,
    args: &Value,
    param_order: &[String],
) -> mlua::Result<Vec<mlua::Value>> {
    let empty = Map::new();
    let args_obj = args.as_object().unwrap_or(&empty);

    let mut call_args = Vec::new();
    let mut consumed: HashSet<&String> = HashSet::new();
    for name in param_order {
        let v = args_obj.get(name).unwrap_or(&Value::Null);
        call_args.push(json_to_lua(lua, v)?);
        consumed.insert(name);
    }
    for (k, v) in args_obj.iter() {
        if !consumed.contains(k) {
            call_args.push(json_to_lua(lua, v)?);
        }
    }
    Ok(call_args)
}

/// A minimal LuaDoc-style comment block: `---` lines accumulate into the
/// description, `-- @param <name> <description>` lines are recorded in
/// order. Drives both a global function's description/params *and* (see
/// [`discover_declarations`]) whether it's exposed as a tool at all: an
/// undocumented global function has no entry here and is treated as a
/// private helper.
#[derive(Default)]
struct ParsedDocComment {
    description: Option<String>,
    params: Vec<(String, String)>,
}

/// Scans raw Lua source for doc-comment blocks immediately preceding a
/// top-level `function <name>(...)` declaration, keyed by that name. Blank
/// lines inside a pending comment block don't break it (comment block, blank
/// line, `function ...` is a common style); any other code line does.
///
/// Only plain `function name(...)` declarations are recognized (not
/// `function tbl.method(...)` or `local function ...`), matching this
/// module's global-function tool contract.
fn parse_doc_comments(source: &str) -> HashMap<String, ParsedDocComment> {
    let mut result = HashMap::new();
    let mut pending_desc: Vec<String> = Vec::new();
    let mut pending_params: Vec<(String, String)> = Vec::new();
    let mut have_pending = false;

    for raw_line in source.lines() {
        let line = raw_line.trim();
        if let Some(rest) = line.strip_prefix("---") {
            pending_desc.push(rest.trim().to_string());
            have_pending = true;
            continue;
        }
        if let Some(rest) = line.strip_prefix("--") {
            let rest = rest.trim();
            if let Some(param_rest) = rest.strip_prefix("@param") {
                let param_rest = param_rest.trim();
                match param_rest.split_once(char::is_whitespace) {
                    Some((name, desc)) => {
                        pending_params.push((name.trim().to_string(), desc.trim().to_string()))
                    }
                    None if !param_rest.is_empty() => {
                        pending_params.push((param_rest.to_string(), String::new()))
                    }
                    None => {}
                }
                have_pending = true;
            }
            continue;
        }
        if let Some(name) = parse_function_decl(line) {
            if have_pending {
                result.insert(
                    name,
                    ParsedDocComment {
                        description: (!pending_desc.is_empty()).then(|| pending_desc.join(" ")),
                        params: std::mem::take(&mut pending_params),
                    },
                );
            }
            pending_desc.clear();
            pending_params.clear();
            have_pending = false;
            continue;
        }
        if line.is_empty() {
            continue;
        }
        pending_desc.clear();
        pending_params.clear();
        have_pending = false;
    }
    result
}

fn parse_function_decl(line: &str) -> Option<String> {
    let rest = line.strip_prefix("function")?;
    let rest = rest.strip_prefix(char::is_whitespace)?;
    let rest = rest.trim_start();
    let end = rest.find(|c: char| !(c.is_alphanumeric() || c == '_'))?;
    if end == 0 {
        return None;
    }
    Some(rest[..end].to_string())
}

/// JSON -> Lua. `null` becomes Lua `nil`; note (this is inherent to Lua, not
/// a bug here) that a `nil` value stored into a table key removes the key --
/// a JSON object field explicitly set to `null` will not show up as a
/// present-but-nil key inside the Lua table passed to a handler that
/// receives a whole table (e.g. via `ctx`, or a schema-less handler's
/// leftover-fields table).
pub fn json_to_lua(lua: &mlua::Lua, value: &Value) -> mlua::Result<mlua::Value> {
    match value {
        Value::Null => Ok(mlua::Value::Nil),
        Value::Bool(b) => Ok(mlua::Value::Boolean(*b)),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(mlua::Value::Integer(i))
            } else if let Some(f) = n.as_f64() {
                Ok(mlua::Value::Number(f))
            } else {
                Err(mlua::Error::RuntimeError(format!(
                    "Unsupported JSON number: {n}"
                )))
            }
        }
        Value::String(s) => Ok(mlua::Value::String(lua.create_string(s)?)),
        Value::Array(arr) => {
            let table = lua.create_table()?;
            for (i, v) in arr.iter().enumerate() {
                table.set((i + 1) as i64, json_to_lua(lua, v)?)?;
            }
            Ok(mlua::Value::Table(table))
        }
        Value::Object(map) => {
            let table = lua.create_table()?;
            for (k, v) in map.iter() {
                table.set(k.as_str(), json_to_lua(lua, v)?)?;
            }
            Ok(mlua::Value::Table(table))
        }
    }
}

/// Lua -> JSON (a tool's return value). A returned function, userdata, or
/// thread is a clear error rather than being silently dropped or coerced.
pub fn lua_to_json(value: mlua::Value) -> Result<Value> {
    match value {
        mlua::Value::Nil => Ok(Value::Null),
        mlua::Value::Boolean(b) => Ok(Value::Bool(b)),
        mlua::Value::Integer(i) => Ok(json!(i)),
        mlua::Value::Number(n) => Ok(json!(n)),
        mlua::Value::String(s) => Ok(Value::String(
            String::from_utf8_lossy(&s.as_bytes()).into_owned(),
        )),
        mlua::Value::Table(t) => lua_table_to_json(t),
        other => bail!(
            "Lua tool returned a {} value, which can't be converted to JSON",
            other.type_name()
        ),
    }
}

/// A table becomes a JSON array exactly when its keys are the contiguous
/// integers `1..=n` for `n = raw_len()` and nothing else -- otherwise (this
/// includes the empty table `{}`, a documented quirk shared with common
/// Lua/JSON bridges like `cjson`/`lunajson`) it becomes a JSON object.
fn lua_table_to_json(table: mlua::Table) -> Result<Value> {
    let len = table.raw_len();
    let mut is_array = len > 0;
    if is_array {
        let mut count = 0usize;
        for pair in table.pairs::<mlua::Value, mlua::Value>() {
            pair.map_err(|e| anyhow!("Failed to iterate Lua table: {e}"))?;
            count += 1;
        }
        is_array = count == len;
    }
    if is_array {
        let mut arr = Vec::with_capacity(len);
        for i in 1..=len as i64 {
            let v: mlua::Value = table
                .get(i)
                .map_err(|e| anyhow!("Failed to read Lua table index {i}: {e}"))?;
            arr.push(lua_to_json(v)?);
        }
        Ok(Value::Array(arr))
    } else {
        let mut map = Map::new();
        for pair in table.pairs::<mlua::Value, mlua::Value>() {
            let (k, v) = pair.map_err(|e| anyhow!("Failed to iterate Lua table: {e}"))?;
            let key = match k {
                mlua::Value::String(s) => String::from_utf8_lossy(&s.as_bytes()).into_owned(),
                mlua::Value::Integer(i) => i.to_string(),
                mlua::Value::Number(n) => n.to_string(),
                other => bail!(
                    "Lua table has a {} key, which can't become a JSON object key",
                    other.type_name()
                ),
            };
            map.insert(key, lua_to_json(v)?);
        }
        Ok(Value::Object(map))
    }
}

/// Builds the `functions.json`-equivalent declaration list for every tool
/// found in `dir`'s `.lua` files -- `register()` entries, plus every
/// *documented* global function (has a `docs.<name>` string and/or a
/// `-- @param`/`---` comment block; undocumented ones are private helpers
/// and are skipped). Called once per config/agent load (see
/// `Functions::init`), not on the per-call dispatch path -- so, unlike
/// [`run_lua_tool`], this runs inline rather than via `spawn_blocking`; see
/// the module docs' "Execution model" section.
///
/// `agent` marks every returned declaration's `FunctionDeclaration::agent`
/// field: `true` for an agent's own directory (dispatched back through that
/// agent, per `extract_call_config_from_agent`), `false` for the global
/// `functions_dir()`.
///
/// Returns declarations in a stable order (`register()` entries in the order
/// they were registered -- i.e. file order, then within-file order --
/// followed by documented global functions in alphabetical order) so the
/// declaration list doesn't reorder from run to run for no reason.
pub fn discover_declarations(dir: &Path, agent: bool) -> Result<Vec<FunctionDeclaration>> {
    let mut lua_files: Vec<PathBuf> = fs::read_dir(dir)
        .with_context(|| format!("Failed to read Lua tools directory {}", dir.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| is_lua_file(p))
        .collect();
    if lua_files.is_empty() {
        return Ok(vec![]);
    }
    lua_files.sort();

    let lua = new_vm();
    let registry: Rc<RefCell<IndexMap<String, RegisteredTool>>> =
        Rc::new(RefCell::new(IndexMap::new()));
    let docs_table = lua
        .create_table()
        .map_err(|e| anyhow!("Failed to create the Lua `docs` table: {e}"))?;
    lua.globals()
        .set("docs", docs_table.clone())
        .map_err(|e| anyhow!("Failed to install the Lua `docs` global: {e}"))?;
    install_register_fn(&lua, registry.clone())
        .map_err(|e| anyhow!("Failed to install the Lua `register` global: {e}"))?;

    let mut doc_comments: HashMap<String, ParsedDocComment> = HashMap::new();
    for path in &lua_files {
        let source = fs::read_to_string(path)
            .with_context(|| format!("Failed to read Lua tool file {}", path.display()))?;
        doc_comments.extend(parse_doc_comments(&source));
        lua.load(source.as_str())
            .set_name(path.display().to_string())
            .exec()
            .map_err(|e| anyhow!("Failed to load Lua tool file {}: {e}", path.display()))?;
    }

    let mut declarations = Vec::new();

    for (name, tool) in registry.borrow().iter() {
        let description = tool.description.clone().unwrap_or_default();
        let params = resolve_param_meta(&lua, &tool.handler, tool.params.as_deref(), None)
            .map_err(|e| anyhow!("Failed to inspect Lua tool '{name}': {e}"))?;
        declarations.push(to_function_declaration(
            name.clone(),
            description,
            params,
            agent,
        ));
    }

    // Every name that could plausibly be a documented global-function tool:
    // anything with a parsed comment block, plus anything with an explicit
    // `docs.<name> = "..."` entry. Sorted for a stable declaration order.
    let mut candidate_names: HashSet<String> = doc_comments.keys().cloned().collect();
    for pair in docs_table.pairs::<mlua::Value, mlua::Value>() {
        let (k, _) = pair.map_err(|e| anyhow!("Failed to read the Lua `docs` table: {e}"))?;
        if let mlua::Value::String(s) = k {
            candidate_names.insert(String::from_utf8_lossy(&s.as_bytes()).into_owned());
        }
    }
    let mut candidate_names: Vec<String> = candidate_names.into_iter().collect();
    candidate_names.sort();

    for name in candidate_names {
        if registry.borrow().contains_key(&name) {
            continue; // register() already declared this tool
        }
        let value: mlua::Value = lua
            .globals()
            .get(name.as_str())
            .map_err(|e| anyhow!("Failed to look up global Lua function '{name}': {e}"))?;
        let mlua::Value::Function(handler) = value else {
            continue; // `docs.<name>`/a comment block with no matching function
        };
        let docs_desc: Option<String> = docs_table
            .get(name.as_str())
            .map_err(|e| anyhow!("Failed to read `docs.{name}`: {e}"))?;
        let comment = doc_comments.get(&name);
        let description = docs_desc.or_else(|| comment.and_then(|c| c.description.clone()));
        let Some(description) = description else {
            continue; // no description anywhere -- a private helper, not a tool
        };
        let comment_params = comment.map(|c| c.params.as_slice());
        let params = resolve_param_meta(&lua, &handler, None, comment_params)
            .map_err(|e| anyhow!("Failed to inspect Lua tool '{name}': {e}"))?;
        declarations.push(to_function_declaration(name, description, params, agent));
    }

    Ok(declarations)
}

fn to_function_declaration(
    name: String,
    description: String,
    params: Vec<ParamMeta>,
    agent: bool,
) -> FunctionDeclaration {
    let mut properties = IndexMap::new();
    let mut required = Vec::new();
    for p in params {
        let schema = JsonSchema {
            type_value: Some(p.type_name.unwrap_or_else(|| "string".to_string())),
            description: p.description,
            properties: None,
            items: None,
            any_of: None,
            enum_value: p.enum_values,
            default: None,
            required: None,
        };
        if p.required {
            required.push(p.name.clone());
        }
        properties.insert(p.name, schema);
    }
    let parameters = JsonSchema {
        type_value: Some("object".to_string()),
        description: None,
        properties: (!properties.is_empty()).then_some(properties),
        items: None,
        any_of: None,
        enum_value: None,
        default: None,
        required: (!required.is_empty()).then_some(required),
    };
    FunctionDeclaration {
        name,
        description,
        parameters,
        agent,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lua() -> mlua::Lua {
        new_vm()
    }

    #[test]
    fn json_to_lua_round_trips_nested_object() {
        let lua = lua();
        let value = json!({
            "a": 1,
            "b": "two",
            "c": [1, 2, 3],
            "d": { "e": true, "f": null },
        });
        let lua_value = json_to_lua(&lua, &value).unwrap();
        let back = lua_to_json(lua_value).unwrap();
        // `d.f: null` is lost on the way in (Lua `nil` removes the table
        // key), so compare against the value with that key already dropped
        // rather than asserting full round-trip equality.
        let expected = json!({
            "a": 1,
            "b": "two",
            "c": [1, 2, 3],
            "d": { "e": true },
        });
        assert_eq!(back, expected);
    }

    #[test]
    fn json_to_lua_round_trips_array() {
        let lua = lua();
        let value = json!([1, "two", [3, 4], { "k": "v" }]);
        let lua_value = json_to_lua(&lua, &value).unwrap();
        let back = lua_to_json(lua_value).unwrap();
        assert_eq!(back, value);
    }

    #[test]
    fn empty_lua_table_becomes_json_object() {
        let lua = lua();
        let table = lua.create_table().unwrap();
        let back = lua_to_json(mlua::Value::Table(table)).unwrap();
        assert_eq!(back, json!({}));
    }

    #[test]
    fn lua_function_return_is_an_error() {
        let lua = lua();
        let f = lua.create_function(|_, ()| Ok(())).unwrap();
        let err = lua_to_json(mlua::Value::Function(f)).unwrap_err();
        assert!(err.to_string().contains("function"));
    }

    #[test]
    fn table_with_non_contiguous_keys_becomes_object_not_array() {
        let lua = lua();
        let table = lua.create_table().unwrap();
        table.set(1, "a").unwrap();
        table.set(3, "c").unwrap();
        let back = lua_to_json(mlua::Value::Table(table)).unwrap();
        assert_eq!(back, json!({"1": "a", "3": "c"}));
    }

    #[test]
    fn signature_param_names_reads_declared_parameter_names() {
        let lua = lua();
        let f: mlua::Function = lua.load("return function(a, b, c) end").eval().unwrap();
        let names = signature_param_names(&lua, &f).unwrap();
        assert_eq!(names, vec!["a", "b", "c"]);
    }

    #[test]
    fn signature_param_names_empty_for_niladic_function() {
        let lua = lua();
        let f: mlua::Function = lua.load("return function() end").eval().unwrap();
        let names = signature_param_names(&lua, &f).unwrap();
        assert!(names.is_empty());
    }

    #[test]
    fn signature_param_names_stops_before_varargs() {
        let lua = lua();
        let f: mlua::Function = lua.load("return function(a, ...) end").eval().unwrap();
        let names = signature_param_names(&lua, &f).unwrap();
        assert_eq!(names, vec!["a"]);
    }

    #[test]
    fn reverse_text_tool_reverses_a_string() {
        // Exactly the `reverse_text.lua` example handed to the user -- no
        // `params`/`@param` annotations at all, so the single `text`
        // parameter is resolved via bare signature introspection. Returns a
        // plain string, not a wrapped table.
        let dir = tempdir();
        std::fs::write(
            dir.path().join("reverse_text.lua"),
            r#"
                docs.reverse_text = "Reverse the characters of a string."
                function reverse_text(text)
                    return text:reverse()
                end
            "#,
        )
        .unwrap();
        let out = run_lua_tool_sync(
            dir.path(),
            "reverse_text",
            json!({"text": "Hello, world!"}),
            HashMap::new(),
        )
        .unwrap()
        .unwrap();
        assert_eq!(out, json!("!dlrow ,olleH"));
    }

    #[test]
    fn get_weather_tool_returns_a_plain_string_with_a_default_param() {
        let dir = tempdir();
        std::fs::write(
            dir.path().join("get_weather.lua"),
            r#"
                --- Get the current weather for a city.
                -- @param location City name.
                -- @param unit Either "celsius" or "fahrenheit".
                function get_weather(location, unit)
                    unit = unit or "celsius"
                    return "It is sunny and 22 degrees " .. unit .. " in " .. location .. "."
                end
            "#,
        )
        .unwrap();
        // No `unit` in the arguments object at all -- the handler's own
        // `unit = unit or "celsius"` default kicks in.
        let out = run_lua_tool_sync(
            dir.path(),
            "get_weather",
            json!({"location": "Paris"}),
            HashMap::new(),
        )
        .unwrap()
        .unwrap();
        assert_eq!(out, json!("It is sunny and 22 degrees celsius in Paris."));
    }

    const WEB_SEARCH_LUA: &str = r##"
        register("web_search", {
            description = "Search the web and return matching titles.",
            params = {
                { name = "query", type = "string", description = "Search query" },
                {
                    name = "category",
                    type = "string",
                    description = "Restrict results to a category",
                    enum = { "news", "images", "videos" },
                    required = false,
                },
                { name = "limit", type = "integer", description = "Max results to return", required = false },
            },
            handler = function(query, category, limit)
                limit = limit or 3
                category = category or "general"
                return "Top " .. limit .. " " .. category .. " results for '" .. query .. "': ..."
            end,
        })
    "##;

    #[test]
    fn register_tool_accepts_rich_param_metadata_and_optional_args() {
        let dir = tempdir();
        std::fs::write(dir.path().join("web_search.lua"), WEB_SEARCH_LUA).unwrap();
        let out = run_lua_tool_sync(
            dir.path(),
            "web_search",
            json!({"query": "rust lang"}),
            HashMap::new(),
        )
        .unwrap()
        .unwrap();
        assert_eq!(out, json!("Top 3 general results for 'rust lang': ..."));
    }

    #[test]
    fn unknown_tool_name_falls_through_as_none() {
        let dir = tempdir();
        std::fs::write(dir.path().join("tool.lua"), "function foo() end").unwrap();
        let out = run_lua_tool_sync(dir.path(), "bar", json!({}), HashMap::new()).unwrap();
        assert!(out.is_none());
    }

    #[test]
    fn no_lua_files_is_not_a_lua_tool_directory() {
        let dir = tempdir();
        assert!(!has_lua_tools(dir.path()));
    }

    #[test]
    fn directory_with_lua_file_is_a_lua_tool_directory() {
        let dir = tempdir();
        std::fs::write(dir.path().join("tool.lua"), "function foo() end").unwrap();
        assert!(has_lua_tools(dir.path()));
    }

    #[test]
    fn lua_tools_dir_for_distinguishes_standalone_and_agent_dispatch() {
        // Matches the *pre-push* `cmd_args` shape at the actual call site in
        // `function.rs`'s `eval` (before `cmd_args.push(json_data.to_string())`
        // runs): empty for a plain/non-agent-flagged call, exactly
        // `[function_name]` for an agent-dispatch (`function.agent == true`)
        // call. Points at `functions_dir()`/`agent_dir()` directly, NOT
        // their `bin/` subdirectory -- that's reserved for external-process
        // tools.
        let standalone = lua_tools_dir_for("reverse_text", &[]);
        assert_eq!(standalone, Config::functions_dir());
        let agent_dispatch = lua_tools_dir_for("todo", &["add_task".into()]);
        assert_eq!(agent_dispatch, Config::agent_dir("todo"));
    }

    #[test]
    fn discover_declarations_on_empty_dir_is_empty() {
        let dir = tempdir();
        let decls = discover_declarations(dir.path(), false).unwrap();
        assert!(decls.is_empty());
    }

    #[test]
    fn discover_skips_undocumented_global_function() {
        let dir = tempdir();
        std::fs::write(dir.path().join("helper.lua"), "function helper(x)\n    return x + 1\nend\n")
            .unwrap();
        let decls = discover_declarations(dir.path(), false).unwrap();
        assert!(decls.is_empty());
    }

    #[test]
    fn discover_finds_documented_global_function_via_docs_table() {
        let dir = tempdir();
        std::fs::write(
            dir.path().join("reverse_text.lua"),
            r#"
                docs.reverse_text = "Reverse the characters of a string."
                function reverse_text(text)
                    return text:reverse()
                end
            "#,
        )
        .unwrap();
        let decls = discover_declarations(dir.path(), false).unwrap();
        assert_eq!(decls.len(), 1);
        let decl = &decls[0];
        assert_eq!(decl.name, "reverse_text");
        assert_eq!(decl.description, "Reverse the characters of a string.");
        assert!(!decl.agent);
        let props = decl.parameters.properties.as_ref().unwrap();
        let text_prop = props.get("text").unwrap();
        assert_eq!(text_prop.type_value.as_deref(), Some("string"));
        assert_eq!(decl.parameters.required.as_deref(), Some(&["text".to_string()][..]));
    }

    #[test]
    fn discover_finds_documented_global_function_via_comment() {
        let dir = tempdir();
        std::fs::write(
            dir.path().join("get_weather.lua"),
            r#"
                --- Get the current weather for a city.
                -- @param location City name, e.g. "Paris".
                -- @param unit Either "celsius" or "fahrenheit".
                function get_weather(location, unit)
                    unit = unit or "celsius"
                    return "..."
                end
            "#,
        )
        .unwrap();
        let decls = discover_declarations(dir.path(), true).unwrap();
        assert_eq!(decls.len(), 1);
        let decl = &decls[0];
        assert_eq!(decl.name, "get_weather");
        assert_eq!(decl.description, "Get the current weather for a city.");
        assert!(decl.agent);
        let props = decl.parameters.properties.as_ref().unwrap();
        assert_eq!(
            props.get("location").unwrap().description.as_deref(),
            Some("City name, e.g. \"Paris\".")
        );
        assert_eq!(
            props.get("unit").unwrap().description.as_deref(),
            Some("Either \"celsius\" or \"fahrenheit\".")
        );
        let mut required = decl.parameters.required.clone().unwrap();
        required.sort();
        assert_eq!(required, vec!["location".to_string(), "unit".to_string()]);
    }

    #[test]
    fn discover_finds_register_tool_with_rich_param_metadata() {
        let dir = tempdir();
        std::fs::write(dir.path().join("web_search.lua"), WEB_SEARCH_LUA).unwrap();
        let decls = discover_declarations(dir.path(), false).unwrap();
        assert_eq!(decls.len(), 1);
        let decl = &decls[0];
        assert_eq!(decl.name, "web_search");
        assert_eq!(
            decl.description,
            "Search the web and return matching titles."
        );
        let props = decl.parameters.properties.as_ref().unwrap();
        assert_eq!(props.len(), 3);
        assert_eq!(props.get("query").unwrap().type_value.as_deref(), Some("string"));
        let category = props.get("category").unwrap();
        assert_eq!(category.type_value.as_deref(), Some("string"));
        assert_eq!(
            category.enum_value.as_deref(),
            Some(&["news".to_string(), "images".to_string(), "videos".to_string()][..])
        );
        assert_eq!(
            props.get("limit").unwrap().type_value.as_deref(),
            Some("integer")
        );
        // Only `query` is required -- `category` and `limit` both say
        // `required = false`.
        assert_eq!(decl.parameters.required.as_deref(), Some(&["query".to_string()][..]));
    }

    #[test]
    fn discover_register_tool_takes_precedence_over_same_named_global_function() {
        let dir = tempdir();
        std::fs::write(
            dir.path().join("reverse_text.lua"),
            r#"
                docs.reverse_text = "Global doc description."
                function reverse_text(text)
                    return text:reverse()
                end

                register("reverse_text", {
                    description = "Register doc description.",
                    handler = function(text)
                        return text:reverse()
                    end,
                })
            "#,
        )
        .unwrap();
        let decls = discover_declarations(dir.path(), false).unwrap();
        assert_eq!(decls.len(), 1);
        assert_eq!(decls[0].description, "Register doc description.");
    }

    /// Minimal temp-dir helper so tests don't need an extra dev-dependency.
    struct TempDir(PathBuf);
    impl TempDir {
        fn path(&self) -> &Path {
            &self.0
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    fn tempdir() -> TempDir {
        let dir = std::env::temp_dir().join(format!(
            "aichat-lua-tool-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        TempDir(dir)
    }
}

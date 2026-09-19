use super::*;

use serde::{Deserialize, Serialize};
use std::fs::read_to_string;

pub type AgentVariables = IndexMap<String, String>;

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct AgentVariable {
    pub name: String,
    pub description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default: Option<String>,
    #[serde(skip_deserializing, default)]
    pub value: String,
}

pub fn list_agents() -> Vec<String> {
    let agents_dir = Config::agents_dir();
    let Ok(entries) = std::fs::read_dir(&agents_dir) else {
        return vec![];
    };
    let mut names: Vec<String> = entries
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().join("index.yaml").exists())
        .filter_map(|entry| entry.file_name().into_string().ok())
        .collect();
    names.sort();
    names
}

pub fn complete_agent_variables(agent_name: &str) -> Vec<(String, Option<String>)> {
    let index_path = Config::agent_dir(agent_name).join("index.yaml");
    if !index_path.exists() {
        return vec![];
    }
    let Ok(content) = read_to_string(&index_path) else {
        return vec![];
    };
    let Ok(role) = serde_yaml::from_str::<Role>(&content) else {
        return vec![];
    };
    role.defined_variables()
        .iter()
        .map(|v| {
            let description = match &v.default {
                Some(default) => format!("{} [default: {default}]", v.description),
                None => v.description.clone(),
            };
            (format!("{}=", v.name), Some(description))
        })
        .collect()
}

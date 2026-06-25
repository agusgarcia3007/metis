//! A real Cortex backend: a small local LLM (e.g. Qwen3) served by a locally-running ollama server.
//! Adds a conversation-aware Chat and a tool-augmented ChatTools on top of the Kernel seam.

use super::{GenerateRequest, Info, Kernel};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::time::Duration;

/// Message is one turn of a conversation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Message {
    pub role: String, // "system" | "user" | "assistant"
    pub content: String,
}

/// The body of a tool: maps JSON arguments to a string result.
pub type ToolFn = Box<dyn Fn(&Value) -> Result<String, String>>;

/// Tool is a function the Cortex can call (a member of the "Hands").
pub struct Tool {
    pub name: String,
    pub description: String,
    pub params: Value, // JSON Schema for the arguments
    pub run: ToolFn,
}

/// OllamaKernel talks to a local ollama server.
pub struct OllamaKernel {
    model: String,
    host: String,
    agent: ureq::Agent,
    pub think: bool, // for thinking models (Qwen3): show chain-of-thought. Off = snappy answers.
}

impl OllamaKernel {
    /// NewOllama returns a kernel backed by the given ollama model and host (default localhost:11434).
    pub fn new(model: &str, host: &str) -> OllamaKernel {
        let host = if host.is_empty() {
            "http://127.0.0.1:11434".to_string()
        } else {
            host.to_string()
        };
        // No global read timeout (Go used Timeout: 0): generations stream for a long time.
        let agent = ureq::AgentBuilder::new().build();
        OllamaKernel {
            model: model.to_string(),
            host,
            agent,
            think: false,
        }
    }

    /// build the ollama `options` object. We cap the thread count when METIS_NUM_THREAD is set:
    /// on a shared PaaS, llama.cpp otherwise spawns one thread per *host* core while the container's
    /// CPU quota is a fraction of that — fine for batched prefill, catastrophic for the per-token
    /// barrier sync in decode. Pinning threads to the real vCPU allocation fixes the decode collapse.
    fn options(&self, temperature: f32) -> Value {
        let mut o = json!({ "temperature": temperature });
        if let Ok(n) = std::env::var("METIS_NUM_THREAD").unwrap_or_default().parse::<i64>() {
            if n > 0 {
                o["num_thread"] = json!(n);
            }
        }
        o
    }

    /// Available reports whether the ollama server is reachable.
    pub fn available(&self) -> bool {
        let resp = self
            .agent
            .get(&format!("{}/api/version", self.host))
            .timeout(Duration::from_secs(3))
            .call();
        matches!(resp, Ok(r) if r.status() == 200)
    }

    /// Chat streams a response to the conversation, invoking on_token for each text chunk.
    pub fn chat(
        &self,
        msgs: &[Message],
        temperature: f32,
        on_token: &mut dyn FnMut(&str),
    ) -> Result<String, String> {
        let body = json!({
            "model": self.model,
            "messages": msgs,
            "stream": true,
            "think": self.think,
            "options": self.options(temperature),
        });
        let resp = self
            .agent
            .post(&format!("{}/api/chat", self.host))
            .set("Content-Type", "application/json")
            .send_json(body);
        let resp = match resp {
            Ok(r) => r,
            Err(ureq::Error::Status(code, r)) => {
                let b = r.into_string().unwrap_or_default();
                return Err(format!("ollama {code}: {}", b.trim()));
            }
            Err(e) => return Err(e.to_string()),
        };

        #[derive(Deserialize, Default)]
        struct ChunkMsg {
            #[serde(default)]
            content: String,
        }
        #[derive(Deserialize, Default)]
        struct Chunk {
            #[serde(default)]
            message: ChunkMsg,
            #[serde(default)]
            done: bool,
        }

        let reader = BufReader::new(resp.into_reader());
        let mut full = String::new();
        for line in reader.lines() {
            let line = match line {
                Ok(l) => l,
                Err(_) => break,
            };
            if line.is_empty() {
                continue;
            }
            let chunk: Chunk = match serde_json::from_str(&line) {
                Ok(c) => c,
                Err(_) => continue,
            };
            if !chunk.message.content.is_empty() {
                full.push_str(&chunk.message.content);
                on_token(&chunk.message.content);
            }
            if chunk.done {
                break;
            }
        }
        Ok(full)
    }

    /// ChatTools runs a tool-augmented conversation: the model may request tool calls, which we
    /// execute and feed back, looping until it produces a final answer. on_event reports tool activity.
    pub fn chat_tools(
        &self,
        msgs: &[Message],
        temperature: f32,
        tools: &[Tool],
        mut on_event: Option<&mut (dyn FnMut(&str) + '_)>,
    ) -> Result<String, String> {
        let mut tool_spec: Vec<Value> = Vec::with_capacity(tools.len());
        let mut by_name: HashMap<&str, &Tool> = HashMap::new();
        for t in tools {
            by_name.insert(t.name.as_str(), t);
            tool_spec.push(json!({
                "type": "function",
                "function": { "name": t.name, "description": t.description, "parameters": t.params },
            }));
        }
        let mut raw: Vec<Value> = msgs
            .iter()
            .map(|m| json!({ "role": m.role, "content": m.content }))
            .collect();

        for _ in 0..6 {
            let req_body = json!({
                "model": self.model, "messages": raw, "tools": tool_spec,
                "stream": false, "think": self.think,
                "options": self.options(temperature),
            });
            let resp = self
                .agent
                .post(&format!("{}/api/chat", self.host))
                .set("Content-Type", "application/json")
                .send_json(req_body);
            let resp = match resp {
                Ok(r) => r,
                Err(ureq::Error::Status(code, r)) => {
                    let b = r.into_string().unwrap_or_default();
                    return Err(format!("ollama {code}: {}", b.trim()));
                }
                Err(e) => return Err(e.to_string()),
            };
            let out: Value = resp.into_json().map_err(|e: std::io::Error| e.to_string())?;
            let message = &out["message"];
            let content = message["content"].as_str().unwrap_or("").to_string();
            let tool_calls = message["tool_calls"].as_array().cloned().unwrap_or_default();
            if tool_calls.is_empty() {
                return Ok(content);
            }
            // echo the assistant tool-call turn back, then append each tool result
            let mut echo: Vec<Value> = Vec::with_capacity(tool_calls.len());
            for tc in &tool_calls {
                let f = &tc["function"];
                echo.push(json!({ "function": { "name": f["name"], "arguments": f["arguments"] } }));
            }
            raw.push(json!({ "role": "assistant", "content": content, "tool_calls": echo }));
            for tc in &tool_calls {
                let f = &tc["function"];
                let name = f["name"].as_str().unwrap_or("");
                let args = &f["arguments"];
                let result = match by_name.get(name) {
                    Some(t) => match (t.run)(args) {
                        Ok(r) => r,
                        Err(e) => format!("error: {e}"),
                    },
                    None => format!("error: unknown tool {name}"),
                };
                if let Some(ev) = on_event.as_mut() {
                    ev(&format!("{name}({args}) = {result}"));
                }
                raw.push(json!({ "role": "tool", "content": result, "tool_name": name }));
            }
        }
        Err("tool loop did not converge".to_string())
    }
}

impl Kernel for OllamaKernel {
    /// Generate satisfies Kernel: a single-prompt completion (wraps Chat with one user message).
    fn generate(
        &self,
        req: GenerateRequest,
        on_token: &mut dyn FnMut(&str),
    ) -> Result<String, String> {
        self.chat(
            &[Message {
                role: "user".to_string(),
                content: req.prompt,
            }],
            req.temperature,
            on_token,
        )
    }

    fn info(&self) -> Info {
        Info {
            backend: "ollama".to_string(),
            model: self.model.clone(),
            ctx_len: 0,
        }
    }

    fn close(&self) -> Result<(), String> {
        Ok(())
    }
}

//! Command metis is the entry point for the Metis system — the CLH-C architecture made real:
//!
//!   Cortex   : a small local LLM (Qwen3 via ollama) — reasoning, in ~1.4 GB
//!   Library  : retrieval over a disk-resident, swappable corpus — knowledge-as-data
//!   Hands    : tools (calc, clock) — exact compute and live data the weights can't hold
//!   Conductor: the loop that retrieves, grounds, calls tools, and answers with citations
//!
//! Subcommands:
//!   index <paths...>   build the Library from .txt/.md files or directories
//!   ask "<question>"   one-shot grounded answer with citations
//!   chat               interactive REPL (grounded if a Library exists; tools always on)
//!   serve | setup | extract | version

use std::io::Write;
use std::path::Path;
use std::time::Duration;

use metis_0::hands;
use metis_0::kernel::{Kernel, Message, OllamaKernel, Tool};
use metis_0::library::{self, Embedder, Extraction, Hit, Store};
use serde_json::json;

const DEFAULT_MODEL: &str = "qwen3:4b";
const EMBED_MODEL: &str = "all-minilm";
const LIB_PATH: &str = "library/index.gob";
const TOP_K: usize = 4;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(|s| s.as_str()).unwrap_or("chat");
    match cmd {
        "version" => println!("Metis 0.2.0 — Cortex + Library (RAG) + Hands, fully local"),
        "index" => run_index(&args[1..]),
        "ask" => run_ask(args[1..].join(" ").trim()),
        "chat" => chat(),
        "serve" => serve(),
        "setup" => ensure_models(),
        "extract" => run_extract_bench(),
        _ => {
            eprintln!("usage: metis [chat | serve | setup | index <paths...> | ask \"<q>\" | version]");
            std::process::exit(2);
        }
    }
}

fn base_url() -> String {
    let h = ollama_host();
    if !h.is_empty() {
        h
    } else {
        "http://127.0.0.1:11434".to_string()
    }
}

/// pullModel asks the ollama server to pull a model (idempotent; fast if already present).
fn pull_model(name: &str) -> Result<(), String> {
    let agent = ureq::AgentBuilder::new()
        .timeout(Duration::from_secs(30 * 60)) // first pull downloads weights
        .build();
    let body = json!({ "model": name, "stream": false });
    let resp = agent
        .post(&format!("{}/api/pull", base_url()))
        .set("Content-Type", "application/json")
        .send_json(body);
    match resp {
        Ok(_) => Ok(()),
        Err(ureq::Error::Status(code, r)) => {
            let b = r.into_string().unwrap_or_default();
            Err(format!("pull {name}: {code} {}", b.trim()))
        }
        Err(e) => Err(format!("pull {name}: {e}")),
    }
}

/// ensureModels waits for ollama, then pulls the Cortex + embedder so the container self-provisions.
fn ensure_models() {
    let k = OllamaKernel::new(&model(), &ollama_host());
    let mut ok = false;
    for _ in 0..60 {
        if k.available() {
            ok = true;
            break;
        }
        eprintln!("waiting for ollama at {} ...", base_url());
        std::thread::sleep(Duration::from_secs(2));
    }
    if !ok {
        eprintln!("ollama not reachable at {}", base_url());
        std::process::exit(1);
    }
    for m in [model(), EMBED_MODEL.to_string()] {
        eprintln!("ensuring model {m:?} (downloading on first run, may take minutes) ...");
        if let Err(e) = pull_model(&m) {
            eprintln!("setup: {e}");
            std::process::exit(1);
        }
        eprintln!("model {m:?} ready");
    }
    eprintln!("setup complete.");
}

fn model() -> String {
    match std::env::var("METIS_MODEL") {
        Ok(m) if !m.is_empty() => m,
        _ => DEFAULT_MODEL.to_string(),
    }
}

/// ollamaHost resolves the ollama base URL from OLLAMA_HOST, or "" to use the local default.
fn ollama_host() -> String {
    let h = std::env::var("OLLAMA_HOST").unwrap_or_default();
    if h.is_empty() {
        return String::new();
    }
    if !h.starts_with("http://") && !h.starts_with("https://") {
        format!("http://{h}")
    } else {
        h
    }
}

// ---- tools (Hands) ----

fn calc_tool() -> Tool {
    Tool {
        name: "calc".to_string(),
        description: "Evaluate an arithmetic expression and return the exact result. Use for any non-trivial math.".to_string(),
        params: json!({
            "type": "object",
            "properties": { "expr": { "type": "string", "description": "e.g. 84937*2261 or (5+3)/2" } },
            "required": ["expr"],
        }),
        run: Box::new(|args| {
            let expr = args["expr"].as_str().unwrap_or("");
            hands::calc(expr)
        }),
    }
}

fn clock_tool() -> Tool {
    Tool {
        name: "current_datetime".to_string(),
        description: "Return the current local date and time. Use when the user asks about today, now, or the date.".to_string(),
        params: json!({ "type": "object", "properties": {} }),
        run: Box::new(|_| hands::now()),
    }
}

fn tools() -> Vec<Tool> {
    vec![calc_tool(), clock_tool()]
}

// ---- Library (index) ----

fn run_index(paths: &[String]) {
    if paths.is_empty() {
        eprintln!("usage: metis index <file-or-dir> [...]");
        std::process::exit(2);
    }
    let files = collect_files(paths);
    if files.is_empty() {
        eprintln!("no .txt/.md files found in: {}", paths.join(" "));
        std::process::exit(1);
    }
    let mut chunks = Vec::new();
    for f in &files {
        let b = match std::fs::read_to_string(f) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("skip {f} : {e}");
                continue;
            }
        };
        let base = Path::new(f)
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| f.clone());
        chunks.extend(library::chunk_text(&b, &base, 120, 30));
    }
    println!("indexing {} files -> {} chunks with {} ...", files.len(), chunks.len(), EMBED_MODEL);

    let emb = Embedder::new(EMBED_MODEL, &ollama_host());
    let st = match library::build(&emb, chunks) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("embed error: {e}");
            eprintln!("hint: ollama serve && ollama pull {EMBED_MODEL}");
            std::process::exit(1);
        }
    };
    if let Some(dir) = Path::new(LIB_PATH).parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    if let Err(e) = st.save(LIB_PATH) {
        eprintln!("save error: {e}");
        std::process::exit(1);
    }
    let size_kb = std::fs::metadata(LIB_PATH).map(|m| m.len() as f64 / 1024.0).unwrap_or(0.0);
    println!(
        "Library built: {} chunks, dim={}, {:.1} KB on disk -> {}",
        st.chunks.len(),
        st.dim,
        size_kb,
        LIB_PATH
    );
    println!("now: metis ask \"...\"   or   metis chat");
}

fn collect_files(paths: &[String]) -> Vec<String> {
    fn keep(p: &Path) -> bool {
        match p.extension().and_then(|e| e.to_str()).map(|e| e.to_lowercase()) {
            Some(e) => e == "txt" || e == "md" || e == "markdown",
            None => false,
        }
    }
    fn walk(dir: &Path, out: &mut Vec<String>) {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.is_dir() {
                    walk(&p, out);
                } else if keep(&p) {
                    out.push(p.to_string_lossy().to_string());
                }
            }
        }
    }
    let mut out = Vec::new();
    for p in paths {
        let path = Path::new(p);
        match std::fs::metadata(path) {
            Ok(info) if info.is_dir() => walk(path, &mut out),
            Ok(_) => out.push(p.clone()), // explicit file: take as-is
            Err(_) => continue,
        }
    }
    out
}

// ---- Conductor: grounded answering ----

const BASE_SYSTEM: &str = "You are Metis, a small, helpful assistant running entirely on local hardware.\n\
TOOL RULES (mandatory, no exceptions):\n\
- For ANY calculation, even one multiplication or division, you MUST call the `calc` tool and use its result. Do NOT compute numbers yourself — you make mistakes. Never write a product/quotient you did not get from `calc`.\n\
- For the current date or time, you MUST call `current_datetime`.\n\
Otherwise be clear, accurate, and concise. Only cite a source when you actually used it for a fact.";

fn rag_system(sources: &str) -> String {
    format!(
        "{BASE_SYSTEM}\n\n\
Answer the user's question using ONLY the numbered SOURCES below when they are relevant, and cite them inline like [1], [2]. \
If the sources do not contain the answer, say so plainly instead of inventing facts.\n\nSOURCES:\n{sources}"
    )
}

/// ground retrieves top-k chunks for the question and returns a system prompt + the hits.
fn ground(store: &Option<Store>, emb: &Embedder, question: &str) -> (String, Vec<Hit>) {
    let store = match store {
        Some(s) if !s.chunks.is_empty() => s,
        _ => return (BASE_SYSTEM.to_string(), Vec::new()),
    };
    let qv = match emb.embed(&[question.to_string()]) {
        Ok(v) if !v.is_empty() => v,
        _ => return (BASE_SYSTEM.to_string(), Vec::new()),
    };
    let hits = store.search(&qv[0], TOP_K);
    // Relevance gate: if nothing is actually similar, don't inject irrelevant sources.
    const MIN_SCORE: f32 = 0.2;
    if hits.is_empty() || hits[0].score < MIN_SCORE {
        return (BASE_SYSTEM.to_string(), Vec::new());
    }
    let mut b = String::new();
    for (i, h) in hits.iter().enumerate() {
        b.push_str(&format!("[{}] ({}) {}\n", i + 1, h.chunk.source, h.chunk.text.trim()));
    }
    (rag_system(&b), hits)
}

fn print_sources(hits: &[Hit]) {
    if hits.is_empty() {
        return;
    }
    print!("\n\x1b[2msources: ");
    for (i, h) in hits.iter().enumerate() {
        if i > 0 {
            print!(", ");
        }
        print!("[{}] {} ({:.2})", i + 1, h.chunk.source, h.score);
    }
    println!("\x1b[0m");
}

fn load_library() -> Option<Store> {
    library::load(LIB_PATH).ok()
}

/// extractGate is the cosine threshold above which the extractive fast path is trusted.
const EXTRACT_GATE: f32 = 0.62;

/// tryExtractive returns a confident extractive answer (fast path) or None to fall back to the LLM.
fn try_extractive(emb: &Embedder, hits: &[Hit], q: &str) -> Option<Extraction> {
    if hits.is_empty() {
        return None;
    }
    match library::extract(emb, hits, q) {
        Ok(ex) if ex.score >= EXTRACT_GATE => Some(ex),
        _ => None,
    }
}

/// runExtractBench measures the extractive fast-path (no LLM) score + latency.
fn run_extract_bench() {
    let emb = Embedder::new(EMBED_MODEL, &ollama_host());
    let store = match load_library() {
        Some(s) => s,
        None => {
            println!("no Library; run: metis index <docs>");
            return;
        }
    };
    let qs = [
        "What does the Zephyrian Protocol mandate about resident memory?",
        "Who ratified the Zephyrian Protocol and in what year?",
        "What is the reference implementation codename?",
        "How many knowledge shards may be cached in RAM?",
        "What is the protocol's mascot?",
        "What is the airspeed velocity of an unladen swallow?", // out-of-domain control
    ];
    println!("== extractive fast-path (no LLM) — score + latency ==\n");
    for q in qs {
        let t0 = std::time::Instant::now();
        let qv = match emb.embed(&[q.to_string()]) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let hits = store.search(&qv[0], TOP_K);
        let ex = library::extract(&emb, &hits, q).unwrap_or_default();
        let ms = t0.elapsed().as_millis();
        println!(
            "Q: {q}\n   answer: {:?}\n   score: {:.2}  latency: {ms}ms  src: {}\n",
            ex.answer, ex.score, ex.source
        );
    }
}

// ---- ask: one-shot grounded answer ----

fn run_ask(question: &str) {
    if question.is_empty() {
        eprintln!("usage: metis ask \"<question>\"");
        std::process::exit(2);
    }
    let k = OllamaKernel::new(&model(), &ollama_host());
    if !k.available() {
        eprintln!("ollama not reachable — run: ollama serve");
        std::process::exit(1);
    }
    let emb = Embedder::new(EMBED_MODEL, &ollama_host());
    let store = load_library();
    let (sys, hits) = ground(&store, &emb, question);

    // CASCADE fast path: a confident extractive lookup answers in ~ms, skipping the LLM entirely.
    if let Some(ex) = try_extractive(&emb, &hits, question) {
        println!("{}", ex.answer);
        println!("\x1b[2msources: [{}] ({:.2}, extractive — no LLM)\x1b[0m", ex.source, ex.score);
        return;
    }

    let msgs = vec![
        Message { role: "system".to_string(), content: sys },
        Message { role: "user".to_string(), content: question.to_string() },
    ];
    let reply = k.chat_tools(
        &msgs,
        0.4,
        &tools(),
        Some(&mut |ev: &str| println!("  \x1b[2m[tool] {ev}\x1b[0m")),
    );
    match reply {
        Ok(r) => {
            println!("{}", r.trim());
            print_sources(&hits);
        }
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}

// ---- chat: interactive, grounded + tools ----

fn chat() {
    let mut k = OllamaKernel::new(&model(), &ollama_host());
    if !k.available() {
        eprintln!("Cortex backend (ollama) not reachable at 127.0.0.1:11434.");
        eprintln!("Start it with:  ollama serve   (then: ollama pull {})", model());
        std::process::exit(1);
    }
    let emb = Embedder::new(EMBED_MODEL, &ollama_host());
    let store = load_library();
    let tools = tools();

    let lib = match &store {
        Some(s) => format!("Library: {} chunks (grounded answers with citations)", s.chunks.len()),
        None => "Library: none (run `metis index <docs>` to ground answers)".to_string(),
    };
    println!("metis chat — Cortex={} + Hands[calc,clock]\n{lib}", model());
    println!("commands: /think  /reset  /exit");
    println!();

    let mut history: Vec<Message> = Vec::new(); // user/assistant turns only; system rebuilt each turn
    let stdin = std::io::stdin();

    print!("you> ");
    let _ = std::io::stdout().flush();
    let mut line = String::new();
    while stdin.read_line(&mut line).unwrap_or(0) > 0 {
        let input = line.trim().to_string();
        line.clear();
        match input.as_str() {
            "" => {
                print!("you> ");
                let _ = std::io::stdout().flush();
                continue;
            }
            "/exit" => return,
            "/reset" => {
                history.clear();
                println!("(history cleared)\nyou> ");
                continue;
            }
            "/think" => {
                k.think = !k.think;
                println!("(reasoning {})\nyou> ", if k.think { "ON" } else { "OFF" });
                continue;
            }
            _ => {}
        }

        let (sys, hits) = ground(&store, &emb, &input);
        // cascade fast path: a confident extractive lookup answers instantly, no LLM
        if let Some(ex) = try_extractive(&emb, &hits, &input) {
            println!("\nmetis> {}\n\x1b[2msources: [{}] ({:.2}, extractive)\x1b[0m", ex.answer, ex.source, ex.score);
            history.push(Message { role: "user".to_string(), content: input.clone() });
            history.push(Message { role: "assistant".to_string(), content: ex.answer });
            print!("\nyou> ");
            let _ = std::io::stdout().flush();
            continue;
        }
        let mut msgs = vec![Message { role: "system".to_string(), content: sys }];
        msgs.extend(history.iter().cloned());
        msgs.push(Message { role: "user".to_string(), content: input.clone() });

        let reply = k.chat_tools(
            &msgs,
            0.2,
            &tools,
            Some(&mut |ev: &str| println!("  \x1b[2m[tool] {ev}\x1b[0m")),
        );
        match reply {
            Ok(r) => {
                println!("\nmetis> {}", r.trim());
                print_sources(&hits);
                history.push(Message { role: "user".to_string(), content: input.clone() });
                history.push(Message { role: "assistant".to_string(), content: r });
            }
            Err(e) => eprintln!("error: {e}"),
        }
        print!("\nyou> ");
        let _ = std::io::stdout().flush();
    }
    let _ = k.close();
}

// ---- serve: minimal HTTP API (for deploying on a VPS) ----

fn serve() {
    let port = std::env::var("PORT").unwrap_or_default();
    let port = if port.is_empty() { "8080".to_string() } else { port };
    let mut k = OllamaKernel::new(&model(), &ollama_host());
    let emb = Embedder::new(EMBED_MODEL, &ollama_host());
    let store = load_library();

    let lib_info = match &store {
        Some(s) => format!("{} chunks", s.chunks.len()),
        None => "no Library".to_string(),
    };
    let server = match tiny_http::Server::http(format!("0.0.0.0:{port}")) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("serve: {e}");
            std::process::exit(1);
        }
    };
    eprintln!(
        "Metis serving on :{port} — Cortex={}, {lib_info}. POST /ask {{\"q\":\"...\"}}",
        model()
    );

    for mut request in server.incoming_requests() {
        let method = request.method().as_str().to_string();
        let url = request.url().to_string();
        let path = url.split('?').next().unwrap_or("").to_string();

        match (method.as_str(), path.as_str()) {
            ("GET", "/healthz") => {
                let _ = request.respond(tiny_http::Response::from_string("ok"));
            }
            ("GET", "/readyz") => {
                if k.available() {
                    let _ = request.respond(tiny_http::Response::from_string("ready"));
                } else {
                    let _ = request.respond(
                        tiny_http::Response::from_string("cortex unavailable").with_status_code(503),
                    );
                }
            }
            ("POST", "/ask") => {
                let mut body = String::new();
                let _ = request.as_reader().read_to_string(&mut body);
                let parsed: serde_json::Value = serde_json::from_str(&body).unwrap_or(json!({}));
                let q = parsed["q"].as_str().unwrap_or("").trim().to_string();
                if q.is_empty() {
                    let resp = tiny_http::Response::from_string(
                        r#"{"error":"body must be {\"q\":\"...\"}"}"#,
                    )
                    .with_status_code(400)
                    .with_header(json_header());
                    let _ = request.respond(resp);
                    continue;
                }
                k.think = parsed["think"].as_bool().unwrap_or(false);
                let (sys, hits) = ground(&store, &emb, &q);

                // cascade fast path: confident extractive lookup, no LLM
                if let Some(ex) = try_extractive(&emb, &hits, &q) {
                    let body = json!({
                        "answer": ex.answer,
                        "path": "extractive",
                        "sources": [{ "n": 1, "source": ex.source, "score": ex.score }],
                    });
                    let resp = tiny_http::Response::from_string(body.to_string())
                        .with_header(json_header());
                    let _ = request.respond(resp);
                    continue;
                }
                let msgs = vec![
                    Message { role: "system".to_string(), content: sys },
                    Message { role: "user".to_string(), content: q.clone() },
                ];
                match k.chat_tools(&msgs, 0.3, &tools(), None) {
                    Ok(answer) => {
                        let sources: Vec<serde_json::Value> = hits
                            .iter()
                            .enumerate()
                            .map(|(i, h)| json!({ "n": i + 1, "source": h.chunk.source, "score": h.score }))
                            .collect();
                        let body = json!({
                            "answer": answer.trim(),
                            "path": "generative",
                            "sources": sources,
                        });
                        let resp = tiny_http::Response::from_string(body.to_string())
                            .with_header(json_header());
                        let _ = request.respond(resp);
                    }
                    Err(e) => {
                        let body = json!({ "error": e }).to_string();
                        let resp = tiny_http::Response::from_string(body)
                            .with_status_code(502)
                            .with_header(json_header());
                        let _ = request.respond(resp);
                    }
                }
            }
            ("POST", _) => {
                let _ = request
                    .respond(tiny_http::Response::from_string("POST only").with_status_code(405));
            }
            _ => {
                let _ = request
                    .respond(tiny_http::Response::from_string("not found").with_status_code(404));
            }
        }
    }
}

fn json_header() -> tiny_http::Header {
    tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap()
}

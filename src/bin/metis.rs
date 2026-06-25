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

use metis_0::conductor::{self, GvsConfig};
use metis_0::hands;
use metis_0::kernel::{Kernel, Message, OllamaKernel, Tool};
use metis_0::library::{self, Chunk, Embedder, Extraction, Hit, Store};
use metis_0::verifier::VerifierKind;
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

/// gvsConfig builds the Generate·Verify·Search settings, overridable from the environment.
///   METIS_SEARCH    = total candidates the loop may try (1 = verify-only, no search). Default 3.
///   METIS_NLI_URL   = URL of the NLI sidecar (e.g. http://nli:9090). Unset → LLM judge.
fn gvs_config() -> GvsConfig {
    let mut c = GvsConfig::default();
    if let Ok(n) = std::env::var("METIS_SEARCH").unwrap_or_default().parse::<u32>() {
        c.max_candidates = n.max(1);
    }
    if let Ok(url) = std::env::var("METIS_NLI_URL") {
        if !url.trim().is_empty() {
            c.verifier = VerifierKind::Nli { url };
        }
    }
    c
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

/// webSearchUrl returns the SearXNG base URL if web augmentation is enabled (METIS_SEARCH_URL).
fn web_search_url() -> Option<String> {
    match std::env::var("METIS_SEARCH_URL") {
        Ok(u) if !u.trim().is_empty() => Some(u),
        _ => None,
    }
}

/// webEvidence queries the live web (SearXNG) and ranks the results against the query with the same
/// CPU embedder used for the local Library, returning them as Hits — "the web as a Library".
fn web_evidence(emb: &Embedder, q: &str) -> Vec<Hit> {
    let base = match web_search_url() {
        Some(u) => u,
        None => return Vec::new(),
    };
    let results = match hands::web_search(&base, q, 8) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("web search failed: {e}");
            return Vec::new();
        }
    };
    if results.is_empty() {
        return Vec::new();
    }
    // Embed query + each result's (title + snippet) in one batch; cosine = dot product (unit vecs).
    let mut texts = Vec::with_capacity(results.len() + 1);
    texts.push(q.to_string());
    for r in &results {
        texts.push(snippet_text(r));
    }
    let vecs = match emb.embed(&texts) {
        Ok(v) if v.len() == texts.len() => v,
        _ => return Vec::new(),
    };
    let qv = &vecs[0];
    let mut hits: Vec<Hit> = results
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let v = &vecs[i + 1];
            let score: f32 = qv.iter().zip(v).map(|(a, b)| a * b).sum();
            Hit {
                chunk: Chunk { text: snippet_text(r), source: r.url.clone(), idx: i, vec: v.clone() },
                score,
            }
        })
        .collect();
    hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    hits
}

fn snippet_text(r: &hands::WebResult) -> String {
    if r.content.trim().is_empty() {
        r.title.clone()
    } else {
        format!("{} — {}", r.title.trim(), r.content.trim())
    }
}

/// research grounds a query in the local Library AND, when the local match is thin and web search is
/// enabled, the live web (SearXNG). Web results are just more evidence: the same GVS loop verifies,
/// cites, and abstains over the combined set — so a tiny Cortex answers open-domain questions
/// without ever trusting the web raw. This is "the web as a swappable Library".
fn research(store: &Option<Store>, emb: &Embedder, q: &str) -> (String, Vec<Hit>) {
    let (sys, mut hits) = ground(store, emb, q);
    // Strong local grounding wins outright; only reach for the web when the Library is thin.
    let local_strong = hits.first().map(|h| h.score >= 0.55).unwrap_or(false);
    if local_strong || web_search_url().is_none() {
        return (sys, hits);
    }
    let web = web_evidence(emb, q);
    if web.is_empty() {
        return (sys, hits);
    }
    hits.extend(web);
    hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    hits.truncate(TOP_K + 2); // a little more room when blending local + web
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
/// Overridable via METIS_EXTRACT_GATE — set it above 1.0 to disable the fast path entirely
/// (every query then goes through Generate·Verify·Search). Default 0.62.
const EXTRACT_GATE_DEFAULT: f32 = 0.62;

fn extract_gate() -> f32 {
    std::env::var("METIS_EXTRACT_GATE")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(EXTRACT_GATE_DEFAULT)
}

/// tryExtractive returns a confident extractive answer (fast path) or None to fall back to the LLM.
fn try_extractive(emb: &Embedder, hits: &[Hit], q: &str) -> Option<Extraction> {
    if hits.is_empty() {
        return None;
    }
    // Type-gate (docs/design/09): the fast-path only answers single-fact lookups. Questions that
    // need comparison, aggregation, a superlative, or cross-chunk chaining must go through GVS — a
    // chunk that merely mentions the entities scores high on cosine but does NOT answer the question
    // (and would bypass the verify gate, risking a fabrication). Measured: this recovers most of the
    // quality the always-on fast-path was costing, while keeping the ~0.1s path for true lookups.
    if library::needs_reasoning(q) {
        return None;
    }
    // Never extractive-shortcut WEB evidence: a snippet/title that echoes the query scores high but
    // is not the answer (e.g. a video title). Web hits must be synthesized and verified, not copied.
    if hits.iter().any(|h| h.chunk.source.starts_with("http")) {
        return None;
    }
    match library::extract(emb, hits, q) {
        Ok(ex) if ex.score >= extract_gate() => Some(ex),
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
    let (sys, hits) = research(&store, &emb, question);

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
    // Generate · Verify · Search: generate, verify against evidence, search if it fails, abstain if nothing holds.
    let tl = tools();
    let mut printer = |ev: &str| println!("  \x1b[2m[{ev}]\x1b[0m");
    match conductor::answer(&k, &msgs, &hits, &tl, &gvs_config(), Some(&mut printer)) {
        Ok(a) => {
            println!("{}", a.text.trim());
            if a.route != conductor::Route::Abstained {
                print_sources(&hits);
            }
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

        let (sys, hits) = research(&store, &emb, &input);
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

        let mut printer = |ev: &str| println!("  \x1b[2m[{ev}]\x1b[0m");
        match conductor::answer(&k, &msgs, &hits, &tools, &gvs_config(), Some(&mut printer)) {
            Ok(a) => {
                println!("\nmetis> {}", a.text.trim());
                if a.route != conductor::Route::Abstained {
                    print_sources(&hits);
                }
                history.push(Message { role: "user".to_string(), content: input.clone() });
                history.push(Message { role: "assistant".to_string(), content: a.text });
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
                let (sys, hits) = research(&store, &emb, &q);

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
                let tl = tools();
                match conductor::answer(&k, &msgs, &hits, &tl, &gvs_config(), None) {
                    Ok(a) => {
                        // Abstained answers cite nothing — they are explicitly *not* grounded.
                        let sources: Vec<serde_json::Value> = if a.route == conductor::Route::Abstained {
                            Vec::new()
                        } else {
                            hits.iter()
                                .enumerate()
                                .map(|(i, h)| json!({ "n": i + 1, "source": h.chunk.source, "score": h.score }))
                                .collect()
                        };
                        let body = json!({
                            "answer": a.text.trim(),
                            "path": a.route.as_str(),
                            "verified": a.verdict == Some(conductor::Verdict::Supported),
                            "attempts": a.attempts,
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

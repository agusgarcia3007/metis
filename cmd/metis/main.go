// Command tinyllm is the entry point for the tiny-llm system — the CLH-C architecture made real:
//
//	Cortex  : a small local LLM (Qwen3-1.7B via ollama) — reasoning, in ~1.4 GB
//	Library : retrieval over a disk-resident, swappable corpus — knowledge-as-data (the research)
//	Hands   : tools (calc, clock) — exact compute and live data the weights can't hold
//	Conductor: the loop that retrieves, grounds, calls tools, and answers with citations
//
// Subcommands:
//
//	index <paths...>   build the Library from .txt/.md files or directories
//	ask "<question>"   one-shot grounded answer with citations
//	chat               interactive REPL (grounded if a Library exists; tools always on)
//	version
package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"runtime/debug"
	"strings"
	"time"

	"github.com/agusgarcia3007/metis/internal/hands"
	"github.com/agusgarcia3007/metis/internal/kernel"
	"github.com/agusgarcia3007/metis/internal/library"
)

const (
	// Qwen3-4B is the Cortex: ~GPT-4o-mini-class on reasoning (AIME 73.8, MMLU 69.7, GPQA 55.9 —
	// vs 1.7B's 48.3/60/40) yet fits a 4 GB box at Q4 (~2.5 GB weights, ~3.2 GB resident at ctx 4096;
	// use ctx 2048 + KV-quant to stay comfortable). Override with METIS_MODEL (e.g. qwen3:1.7b for
	// max speed on a tight box).
	defaultModel = "qwen3:4b"
	embedModel   = "all-minilm"
	libPath      = "library/index.gob"
	topK         = 4
)

func main() {
	// Stay lean: the model's RAM lives in the ollama process; keep the Go heap tiny.
	debug.SetGCPercent(30)
	debug.SetMemoryLimit(512 << 20)

	args := os.Args[1:]
	cmd := "chat"
	if len(args) > 0 {
		cmd = args[0]
	}
	switch cmd {
	case "version":
		fmt.Println("Metis 0.2.0 — Cortex + Library (RAG) + Hands, fully local")
	case "index":
		runIndex(args[1:])
	case "ask":
		runAsk(strings.TrimSpace(strings.Join(args[1:], " ")))
	case "chat":
		chat()
	case "serve":
		serve()
	case "setup":
		ensureModels()
	case "extract":
		runExtractBench()
	default:
		fmt.Fprintf(os.Stderr, "usage: metis [chat | serve | setup | index <paths...> | ask \"<q>\" | version]\n")
		os.Exit(2)
	}
}

func baseURL() string {
	if h := ollamaHost(); h != "" {
		return h
	}
	return "http://127.0.0.1:11434"
}

// pullModel asks the ollama server to pull a model (idempotent; fast if already present).
func pullModel(ctx context.Context, name string) error {
	body, _ := json.Marshal(map[string]any{"model": name, "stream": false})
	req, _ := http.NewRequestWithContext(ctx, "POST", baseURL()+"/api/pull", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	client := &http.Client{Timeout: 30 * time.Minute} // first pull downloads weights
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		b, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("pull %s: %d %s", name, resp.StatusCode, strings.TrimSpace(string(b)))
	}
	io.Copy(io.Discard, resp.Body)
	return nil
}

// ensureModels waits for ollama, then pulls the Cortex + embedder so the container self-provisions.
func ensureModels() {
	k := kernel.NewOllama(model(), ollamaHost())
	for i := 0; i < 60 && !k.Available(); i++ {
		log.Printf("waiting for ollama at %s ...", baseURL())
		time.Sleep(2 * time.Second)
	}
	if !k.Available() {
		log.Fatalf("ollama not reachable at %s", baseURL())
	}
	for _, m := range []string{model(), embedModel} {
		log.Printf("ensuring model %q (downloading on first run, may take minutes) ...", m)
		if err := pullModel(context.Background(), m); err != nil {
			log.Fatalf("setup: %v", err)
		}
		log.Printf("model %q ready", m)
	}
	log.Println("setup complete.")
}

func model() string {
	if m := os.Getenv("METIS_MODEL"); m != "" {
		return m
	}
	return defaultModel
}

// ollamaHost resolves the ollama base URL from OLLAMA_HOST (for Docker/remote), or "" to use the
// local default (127.0.0.1:11434).
func ollamaHost() string {
	h := os.Getenv("OLLAMA_HOST")
	if h == "" {
		return ""
	}
	if !strings.HasPrefix(h, "http://") && !strings.HasPrefix(h, "https://") {
		h = "http://" + h
	}
	return h
}

// ---- tools (Hands) ----

var calcTool = kernel.Tool{
	Name:        "calc",
	Description: "Evaluate an arithmetic expression and return the exact result. Use for any non-trivial math.",
	Params: map[string]any{
		"type":       "object",
		"properties": map[string]any{"expr": map[string]any{"type": "string", "description": "e.g. 84937*2261 or (5+3)/2"}},
		"required":   []string{"expr"},
	},
	Run: func(args map[string]any) (string, error) {
		expr, _ := args["expr"].(string)
		return hands.Calc(expr)
	},
}

var clockTool = kernel.Tool{
	Name:        "current_datetime",
	Description: "Return the current local date and time. Use when the user asks about today, now, or the date.",
	Params:      map[string]any{"type": "object", "properties": map[string]any{}},
	Run:         func(map[string]any) (string, error) { return hands.Now() },
}

// ---- Library (index) ----

func runIndex(paths []string) {
	if len(paths) == 0 {
		fmt.Fprintln(os.Stderr, "usage: metis index <file-or-dir> [...]")
		os.Exit(2)
	}
	files := collectFiles(paths)
	if len(files) == 0 {
		fmt.Fprintln(os.Stderr, "no .txt/.md files found in:", strings.Join(paths, " "))
		os.Exit(1)
	}
	var chunks []library.Chunk
	for _, f := range files {
		b, err := os.ReadFile(f)
		if err != nil {
			fmt.Fprintln(os.Stderr, "skip", f, ":", err)
			continue
		}
		chunks = append(chunks, library.ChunkText(string(b), filepath.Base(f), 120, 30)...)
	}
	fmt.Printf("indexing %d files -> %d chunks with %s ...\n", len(files), len(chunks), embedModel)

	emb := library.NewEmbedder(embedModel, ollamaHost())
	st, err := library.Build(context.Background(), emb, chunks)
	if err != nil {
		fmt.Fprintln(os.Stderr, "embed error:", err)
		fmt.Fprintln(os.Stderr, "hint: ollama serve && ollama pull "+embedModel)
		os.Exit(1)
	}
	_ = os.MkdirAll(filepath.Dir(libPath), 0o755)
	if err := st.Save(libPath); err != nil {
		fmt.Fprintln(os.Stderr, "save error:", err)
		os.Exit(1)
	}
	fi, _ := os.Stat(libPath)
	fmt.Printf("Library built: %d chunks, dim=%d, %.1f KB on disk -> %s\n", len(st.Chunks), st.Dim, float64(fi.Size())/1024, libPath)
	fmt.Println("now: metis ask \"...\"   or   metis chat")
}

func collectFiles(paths []string) []string {
	var out []string
	keep := func(p string) bool {
		e := strings.ToLower(filepath.Ext(p))
		return e == ".txt" || e == ".md" || e == ".markdown"
	}
	for _, p := range paths {
		info, err := os.Stat(p)
		if err != nil {
			continue
		}
		if info.IsDir() {
			filepath.WalkDir(p, func(path string, d os.DirEntry, err error) error {
				if err == nil && !d.IsDir() && keep(path) {
					out = append(out, path)
				}
				return nil
			})
		} else {
			out = append(out, p) // explicit file: take as-is
		}
	}
	return out
}

// ---- Conductor: grounded answering ----

const baseSystem = "You are Metis, a small, helpful assistant running entirely on local hardware.\n" +
	"TOOL RULES (mandatory, no exceptions):\n" +
	"- For ANY calculation, even one multiplication or division, you MUST call the `calc` tool and use its result. " +
	"Do NOT compute numbers yourself — you make mistakes. Never write a product/quotient you did not get from `calc`.\n" +
	"- For the current date or time, you MUST call `current_datetime`.\n" +
	"Otherwise be clear, accurate, and concise. Only cite a source when you actually used it for a fact."

const ragSystem = baseSystem + "\n\n" +
	"Answer the user's question using ONLY the numbered SOURCES below when they are relevant, and cite them inline like [1], [2]. " +
	"If the sources do not contain the answer, say so plainly instead of inventing facts.\n\nSOURCES:\n%s"

// ground retrieves top-k chunks for the question and returns a system prompt + the hits (or the base
// system prompt and nil if no Library exists).
func ground(ctx context.Context, store *library.Store, emb *library.Embedder, question string) (string, []library.Hit) {
	if store == nil || len(store.Chunks) == 0 {
		return baseSystem, nil
	}
	qv, err := emb.Embed(ctx, []string{question})
	if err != nil || len(qv) == 0 {
		return baseSystem, nil
	}
	hits := store.Search(qv[0], topK)
	// Relevance gate: if nothing is actually similar, don't inject (irrelevant) sources — this stops
	// spurious citations on questions the Library can't answer (e.g. arithmetic).
	const minScore = 0.2
	if len(hits) == 0 || hits[0].Score < minScore {
		return baseSystem, nil
	}
	var b strings.Builder
	for i, h := range hits {
		fmt.Fprintf(&b, "[%d] (%s) %s\n", i+1, h.Source, strings.TrimSpace(h.Text))
	}
	return fmt.Sprintf(ragSystem, b.String()), hits
}

func printSources(hits []library.Hit) {
	if len(hits) == 0 {
		return
	}
	fmt.Printf("\n\033[2msources: ")
	for i, h := range hits {
		if i > 0 {
			fmt.Print(", ")
		}
		fmt.Printf("[%d] %s (%.2f)", i+1, h.Source, h.Score)
	}
	fmt.Println("\033[0m")
}

func loadLibrary() *library.Store {
	st, err := library.Load(libPath)
	if err != nil {
		return nil
	}
	return st
}

// runExtractBench measures the extractive fast-path (no LLM): for each question it retrieves, extracts
// the best sentence, and reports score + latency — to calibrate the cascade's confidence gate.
func runExtractBench() {
	emb := library.NewEmbedder(embedModel, ollamaHost())
	store := loadLibrary()
	if store == nil {
		fmt.Println("no Library; run: metis index <docs>")
		return
	}
	qs := []string{
		"What does the Zephyrian Protocol mandate about resident memory?",
		"Who ratified the Zephyrian Protocol and in what year?",
		"What is the reference implementation codename?",
		"How many knowledge shards may be cached in RAM?",
		"What is the protocol's mascot?",
		"What is the airspeed velocity of an unladen swallow?", // out-of-domain control
	}
	fmt.Printf("== extractive fast-path (no LLM) — score + latency ==\n\n")
	for _, q := range qs {
		t0 := time.Now()
		qv, _ := emb.Embed(context.Background(), []string{q})
		hits := store.Search(qv[0], topK)
		ex, _ := library.Extract(context.Background(), emb, hits, q)
		ms := time.Since(t0).Milliseconds()
		fmt.Printf("Q: %s\n   answer: %q\n   score: %.2f  latency: %dms  src: %s\n\n", q, ex.Answer, ex.Score, ms, ex.Source)
	}
}

// ---- ask: one-shot grounded answer ----

func runAsk(question string) {
	if question == "" {
		fmt.Fprintln(os.Stderr, "usage: metis ask \"<question>\"")
		os.Exit(2)
	}
	k := kernel.NewOllama(model(), ollamaHost())
	if !k.Available() {
		fmt.Fprintln(os.Stderr, "ollama not reachable — run: ollama serve")
		os.Exit(1)
	}
	emb := library.NewEmbedder(embedModel, ollamaHost())
	store := loadLibrary()
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()
	sys, hits := ground(ctx, store, emb, question)

	// CASCADE fast path: a confident extractive lookup answers in ~ms, skipping the LLM entirely.
	if ex, ok := tryExtractive(ctx, emb, hits, question); ok {
		fmt.Println(ex.Answer)
		fmt.Printf("\033[2msources: [%s] (%.2f, extractive — no LLM)\033[0m\n", ex.Source, ex.Score)
		return
	}

	msgs := []kernel.Message{{Role: "system", Content: sys}, {Role: "user", Content: question}}
	reply, err := k.ChatTools(ctx, msgs, 0.4, []kernel.Tool{calcTool, clockTool},
		func(ev string) { fmt.Printf("  \033[2m[tool] %s\033[0m\n", ev) })
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
	fmt.Println(strings.TrimSpace(reply))
	printSources(hits)
}

// extractGate is the cosine threshold above which the extractive fast path is trusted (calibrated:
// in-domain factoids score 0.6–0.9; off-topic ~0.17). Conservative so quality stays high.
const extractGate = 0.62

// tryExtractive returns a confident extractive answer (fast path) or ok=false to fall back to the LLM.
func tryExtractive(ctx context.Context, emb *library.Embedder, hits []library.Hit, q string) (library.Extraction, bool) {
	if len(hits) == 0 {
		return library.Extraction{}, false
	}
	ex, err := library.Extract(ctx, emb, hits, q)
	if err != nil || ex.Score < extractGate {
		return library.Extraction{}, false
	}
	return ex, true
}

// ---- chat: interactive, grounded + tools ----

func chat() {
	k := kernel.NewOllama(model(), ollamaHost())
	defer k.Close()
	if !k.Available() {
		fmt.Fprintln(os.Stderr, "Cortex backend (ollama) not reachable at 127.0.0.1:11434.")
		fmt.Fprintln(os.Stderr, "Start it with:  ollama serve   (then: ollama pull "+model()+")")
		os.Exit(1)
	}
	emb := library.NewEmbedder(embedModel, ollamaHost())
	store := loadLibrary()
	tools := []kernel.Tool{calcTool, clockTool}

	lib := "Library: none (run `metis index <docs>` to ground answers)"
	if store != nil {
		lib = fmt.Sprintf("Library: %d chunks (grounded answers with citations)", len(store.Chunks))
	}
	fmt.Printf("metis chat — Cortex=%s + Hands[calc,clock]\n%s\n", model(), lib)
	fmt.Println("commands: /think  /reset  /exit")
	fmt.Println()

	history := []kernel.Message{} // user/assistant turns only; system is rebuilt each turn
	sc := bufio.NewScanner(os.Stdin)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	fmt.Print("you> ")
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		switch line {
		case "":
			fmt.Print("you> ")
			continue
		case "/exit":
			return
		case "/reset":
			history = history[:0]
			fmt.Println("(history cleared)\nyou> ")
			continue
		case "/think":
			k.Think = !k.Think
			fmt.Printf("(reasoning %s)\nyou> ", map[bool]string{true: "ON", false: "OFF"}[k.Think])
			continue
		}

		ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
		sys, hits := ground(ctx, store, emb, line)
		// cascade fast path: a confident extractive lookup answers instantly, no LLM
		if ex, ok := tryExtractive(ctx, emb, hits, line); ok {
			cancel()
			fmt.Printf("\nmetis> %s\n\033[2msources: [%s] (%.2f, extractive)\033[0m\n", ex.Answer, ex.Source, ex.Score)
			history = append(history,
				kernel.Message{Role: "user", Content: line},
				kernel.Message{Role: "assistant", Content: ex.Answer})
			fmt.Print("\nyou> ")
			continue
		}
		msgs := append([]kernel.Message{{Role: "system", Content: sys}}, history...)
		msgs = append(msgs, kernel.Message{Role: "user", Content: line})

		reply, err := k.ChatTools(ctx, msgs, 0.2, tools,
			func(ev string) { fmt.Printf("  \033[2m[tool] %s\033[0m\n", ev) })
		cancel()
		if err != nil && ctx.Err() == nil {
			fmt.Fprintln(os.Stderr, "error:", err)
		} else {
			fmt.Printf("\nmetis> %s\n", strings.TrimSpace(reply))
			printSources(hits)
			history = append(history,
				kernel.Message{Role: "user", Content: line},
				kernel.Message{Role: "assistant", Content: reply})
		}
		fmt.Print("\nyou> ")
	}
}

// ---- serve: minimal HTTP API (for deploying on a VPS) ----

type askResponse struct {
	Answer  string   `json:"answer"`
	Path    string   `json:"path,omitempty"` // "extractive" (fast, no LLM) | "generative"
	Sources []source `json:"sources,omitempty"`
}
type source struct {
	N      int     `json:"n"`
	Source string  `json:"source"`
	Score  float32 `json:"score"`
}

func serve() {
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}
	k := kernel.NewOllama(model(), ollamaHost())
	emb := library.NewEmbedder(embedModel, ollamaHost())
	store := loadLibrary()

	mux := http.NewServeMux()
	// liveness (always ok once the process is up)
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) { fmt.Fprint(w, "ok") })
	// readiness (ok once the Cortex backend is reachable)
	mux.HandleFunc("/readyz", func(w http.ResponseWriter, r *http.Request) {
		if k.Available() {
			fmt.Fprint(w, "ready")
			return
		}
		http.Error(w, "cortex unavailable", http.StatusServiceUnavailable)
	})
	// POST /ask  {"q":"...","think":false} -> {"answer":"...","sources":[...]}
	mux.HandleFunc("/ask", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "POST only", http.StatusMethodNotAllowed)
			return
		}
		var req struct {
			Q     string `json:"q"`
			Think bool   `json:"think"`
		}
		if json.NewDecoder(r.Body).Decode(&req) != nil || strings.TrimSpace(req.Q) == "" {
			http.Error(w, `{"error":"body must be {\"q\":\"...\"}"}`, http.StatusBadRequest)
			return
		}
		k.Think = req.Think
		ctx := r.Context()
		sys, hits := ground(ctx, store, emb, req.Q)
		// cascade fast path: confident extractive lookup, no LLM
		if ex, ok := tryExtractive(ctx, emb, hits, req.Q); ok {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(askResponse{Answer: ex.Answer, Path: "extractive",
				Sources: []source{{N: 1, Source: ex.Source, Score: ex.Score}}})
			return
		}
		msgs := []kernel.Message{{Role: "system", Content: sys}, {Role: "user", Content: req.Q}}
		answer, err := k.ChatTools(ctx, msgs, 0.3, []kernel.Tool{calcTool, clockTool}, nil)
		if err != nil {
			http.Error(w, fmt.Sprintf(`{"error":%q}`, err.Error()), http.StatusBadGateway)
			return
		}
		resp := askResponse{Answer: strings.TrimSpace(answer), Path: "generative"}
		for i, h := range hits {
			resp.Sources = append(resp.Sources, source{N: i + 1, Source: h.Source, Score: h.Score})
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	})

	libInfo := "no Library"
	if store != nil {
		libInfo = fmt.Sprintf("%d chunks", len(store.Chunks))
	}
	log.Printf("Metis serving on :%s — Cortex=%s, %s. POST /ask {\"q\":\"...\"}", port, model(), libInfo)
	log.Fatal(http.ListenAndServe(":"+port, mux))
}

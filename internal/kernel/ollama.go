package kernel

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// OllamaKernel is a real Cortex backend: it runs a small local LLM (e.g. Qwen3-1.7B) through a
// locally-running ollama server (which bundles ggml — the Go+ggml pattern from research 05). It
// satisfies kernel.Kernel and adds a conversation-aware Chat method for the REPL.
type OllamaKernel struct {
	model  string
	host   string
	client *http.Client
	Think  bool // for thinking models (Qwen3): show chain-of-thought. Off = snappy answers.
}

// Message is one turn of a conversation.
type Message struct {
	Role    string `json:"role"` // "system" | "user" | "assistant"
	Content string `json:"content"`
}

// NewOllama returns a kernel backed by the given ollama model and host (default localhost:11434).
func NewOllama(model, host string) *OllamaKernel {
	if host == "" {
		host = "http://127.0.0.1:11434"
	}
	return &OllamaKernel{model: model, host: host, client: &http.Client{Timeout: 0}}
}

// Available reports whether the ollama server is reachable.
func (k *OllamaKernel) Available() bool {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, "GET", k.host+"/api/version", nil)
	resp, err := k.client.Do(req)
	if err != nil {
		return false
	}
	resp.Body.Close()
	return resp.StatusCode == 200
}

// Chat streams a response to the conversation, invoking onToken for each text chunk.
func (k *OllamaKernel) Chat(ctx context.Context, msgs []Message, temperature float32, onToken func(string)) (string, error) {
	body, _ := json.Marshal(map[string]any{
		"model":    k.model,
		"messages": msgs,
		"stream":   true,
		"think":    k.Think,
		"options":  map[string]any{"temperature": temperature},
	})
	req, err := http.NewRequestWithContext(ctx, "POST", k.host+"/api/chat", bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := k.client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		b, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("ollama %d: %s", resp.StatusCode, strings.TrimSpace(string(b)))
	}
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	var full strings.Builder
	for sc.Scan() {
		line := sc.Bytes()
		if len(line) == 0 {
			continue
		}
		var chunk struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
			Done bool `json:"done"`
		}
		if json.Unmarshal(line, &chunk) != nil {
			continue
		}
		if chunk.Message.Content != "" {
			full.WriteString(chunk.Message.Content)
			onToken(chunk.Message.Content)
		}
		if chunk.Done {
			break
		}
	}
	return full.String(), sc.Err()
}

// Tool is a function the Cortex can call (a member of the "Hands").
type Tool struct {
	Name        string
	Description string
	Params      map[string]any // JSON Schema for the arguments
	Run         func(args map[string]any) (string, error)
}

// ChatTools runs a tool-augmented conversation: the model may request tool calls, which we execute
// and feed back, looping until it produces a final answer. onEvent reports tool activity. This is how
// a small Cortex offloads its weaknesses (exact arithmetic, live data) to deterministic code.
func (k *OllamaKernel) ChatTools(ctx context.Context, msgs []Message, temperature float32, tools []Tool, onEvent func(string)) (string, error) {
	toolSpec := make([]map[string]any, 0, len(tools))
	byName := map[string]Tool{}
	for _, t := range tools {
		byName[t.Name] = t
		toolSpec = append(toolSpec, map[string]any{
			"type":     "function",
			"function": map[string]any{"name": t.Name, "description": t.Description, "parameters": t.Params},
		})
	}
	raw := make([]map[string]any, 0, len(msgs)+8)
	for _, m := range msgs {
		raw = append(raw, map[string]any{"role": m.Role, "content": m.Content})
	}
	for iter := 0; iter < 6; iter++ {
		reqBody, _ := json.Marshal(map[string]any{
			"model": k.model, "messages": raw, "tools": toolSpec,
			"stream": false, "think": k.Think,
			"options": map[string]any{"temperature": temperature},
		})
		req, err := http.NewRequestWithContext(ctx, "POST", k.host+"/api/chat", bytes.NewReader(reqBody))
		if err != nil {
			return "", err
		}
		req.Header.Set("Content-Type", "application/json")
		resp, err := k.client.Do(req)
		if err != nil {
			return "", err
		}
		var out struct {
			Message struct {
				Content   string `json:"content"`
				ToolCalls []struct {
					Function struct {
						Name      string         `json:"name"`
						Arguments map[string]any `json:"arguments"`
					} `json:"function"`
				} `json:"tool_calls"`
			} `json:"message"`
		}
		derr := json.NewDecoder(resp.Body).Decode(&out)
		resp.Body.Close()
		if derr != nil {
			return "", derr
		}
		if len(out.Message.ToolCalls) == 0 {
			return out.Message.Content, nil
		}
		// echo the assistant tool-call turn back, then append each tool result
		echo := make([]map[string]any, 0, len(out.Message.ToolCalls))
		for _, tc := range out.Message.ToolCalls {
			echo = append(echo, map[string]any{"function": map[string]any{"name": tc.Function.Name, "arguments": tc.Function.Arguments}})
		}
		raw = append(raw, map[string]any{"role": "assistant", "content": out.Message.Content, "tool_calls": echo})
		for _, tc := range out.Message.ToolCalls {
			var result string
			if t, ok := byName[tc.Function.Name]; ok {
				if r, err := t.Run(tc.Function.Arguments); err != nil {
					result = "error: " + err.Error()
				} else {
					result = r
				}
			} else {
				result = "error: unknown tool " + tc.Function.Name
			}
			if onEvent != nil {
				onEvent(fmt.Sprintf("%s(%v) = %s", tc.Function.Name, tc.Function.Arguments, result))
			}
			raw = append(raw, map[string]any{"role": "tool", "content": result, "tool_name": tc.Function.Name})
		}
	}
	return "", fmt.Errorf("tool loop did not converge")
}

// Generate satisfies kernel.Kernel: a single-prompt completion (wraps Chat with one user message).
func (k *OllamaKernel) Generate(ctx context.Context, req GenerateRequest, onToken func(string)) (string, error) {
	return k.Chat(ctx, []Message{{Role: "user", Content: req.Prompt}}, req.Temperature, onToken)
}

// Info reports the backend/model.
func (k *OllamaKernel) Info() Info { return Info{Backend: "ollama", Model: k.model, CtxLen: 0} }

// Close is a no-op (the ollama server is managed externally).
func (k *OllamaKernel) Close() error { return nil }

var _ Kernel = (*OllamaKernel)(nil)

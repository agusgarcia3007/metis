// Package library is tiny-llm's knowledge plane: the "Library" of the CLH-C architecture. It turns a
// corpus of documents into a disk-resident, swappable index that a small Cortex retrieves from at
// query time. This is the research thesis made real (docs/research/04): most of a big model's
// parameters memorize facts; we move that knowledge OUT of the weights into data on disk, so a tiny
// reasoner + retrieval can rival a much larger parametric model — and stay portable.
package library

import (
	"bytes"
	"context"
	"encoding/gob"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"os"
	"sort"
	"strings"
	"time"
)

// Chunk is one retrievable unit of knowledge with its embedding.
type Chunk struct {
	Text   string
	Source string
	Idx    int
	Vec    []float32
}

// Store is the on-disk knowledge index (knowledge-as-data: swap the file, swap the brain's facts).
type Store struct {
	Model  string
	Dim    int
	Chunks []Chunk
}

// Hit is a retrieved chunk with its similarity score.
type Hit struct {
	Chunk
	Score float32
}

// Embedder calls a local ollama embedding model.
type Embedder struct {
	Model  string
	host   string
	client *http.Client
}

// NewEmbedder returns an embedder backed by the given ollama model (e.g. "all-minilm").
func NewEmbedder(model, host string) *Embedder {
	if host == "" {
		host = "http://127.0.0.1:11434"
	}
	return &Embedder{Model: model, host: host, client: &http.Client{Timeout: 120 * time.Second}}
}

// Embed returns one vector per input text.
func (e *Embedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	body, _ := json.Marshal(map[string]any{"model": e.Model, "input": texts})
	req, err := http.NewRequestWithContext(ctx, "POST", e.host+"/api/embed", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := e.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("embed: ollama %d (is `ollama pull %s` done?)", resp.StatusCode, e.Model)
	}
	var out struct {
		Embeddings [][]float32 `json:"embeddings"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	if len(out.Embeddings) != len(texts) {
		return nil, fmt.Errorf("embed: got %d vectors for %d texts", len(out.Embeddings), len(texts))
	}
	for i := range out.Embeddings {
		normalize(out.Embeddings[i])
	}
	return out.Embeddings, nil
}

func normalize(v []float32) {
	var n float64
	for _, x := range v {
		n += float64(x) * float64(x)
	}
	n = math.Sqrt(n)
	if n == 0 {
		return
	}
	inv := float32(1 / n)
	for i := range v {
		v[i] *= inv
	}
}

func cosine(a, b []float32) float32 {
	// vectors are unit-normalized at ingest/query, so dot product == cosine
	var s float32
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

// Chunk splits text into ~size-word windows with overlap, tagged with the source.
func ChunkText(text, source string, size, overlap int) []Chunk {
	fields := strings.Fields(text)
	if len(fields) == 0 {
		return nil
	}
	if size <= 0 {
		size = 120
	}
	if overlap < 0 || overlap >= size {
		overlap = size / 4
	}
	var chunks []Chunk
	for start, idx := 0, 0; start < len(fields); start += size - overlap {
		end := start + size
		if end > len(fields) {
			end = len(fields)
		}
		chunks = append(chunks, Chunk{Text: strings.Join(fields[start:end], " "), Source: source, Idx: idx})
		idx++
		if end == len(fields) {
			break
		}
	}
	return chunks
}

// Build embeds the given chunks (in batches) into a Store.
func Build(ctx context.Context, emb *Embedder, chunks []Chunk) (*Store, error) {
	const batch = 32
	st := &Store{Model: emb.Model}
	for i := 0; i < len(chunks); i += batch {
		j := i + batch
		if j > len(chunks) {
			j = len(chunks)
		}
		texts := make([]string, j-i)
		for k := range texts {
			texts[k] = chunks[i+k].Text
		}
		vecs, err := emb.Embed(ctx, texts)
		if err != nil {
			return nil, err
		}
		for k := range vecs {
			c := chunks[i+k]
			c.Vec = vecs[k]
			st.Chunks = append(st.Chunks, c)
		}
	}
	if len(st.Chunks) > 0 {
		st.Dim = len(st.Chunks[0].Vec)
	}
	return st, nil
}

// Search returns the top-k chunks most similar to the query embedding.
func (st *Store) Search(qVec []float32, k int) []Hit {
	hits := make([]Hit, 0, len(st.Chunks))
	for _, c := range st.Chunks {
		hits = append(hits, Hit{Chunk: c, Score: cosine(qVec, c.Vec)})
	}
	sort.Slice(hits, func(i, j int) bool { return hits[i].Score > hits[j].Score })
	if k < len(hits) {
		hits = hits[:k]
	}
	return hits
}

// Save/Load persist the index to disk (this file IS the swappable knowledge).
func (st *Store) Save(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return gob.NewEncoder(f).Encode(st)
}

func Load(path string) (*Store, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var st Store
	if err := gob.NewDecoder(f).Decode(&st); err != nil {
		return nil, err
	}
	return &st, nil
}

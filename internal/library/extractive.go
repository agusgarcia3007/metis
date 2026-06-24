package library

import (
	"context"
	"regexp"
	"strings"
)

// extractive.go is the cascade's fast path: for a factual "lookup" query whose answer is a span
// already in a retrieved chunk, score the chunk's sentences against the query with the (tiny, CPU)
// embedder and return the best one — WITHOUT running the generative LLM. Research (research 10):
// generation is ~75% of RAG latency; a reader-free extractive step (DensePhrases ~74 ms CPU) ties
// full RAG on single-hop factoids. This is the single biggest latency lever for the common case.

var sentenceSplit = regexp.MustCompile(`(?:[.!?:;]|\n)+\s*`)

// SplitSentences breaks text into candidate answer spans (sentences/clauses), dropping trivia.
func SplitSentences(text string) []string {
	var out []string
	for _, p := range sentenceSplit.Split(text, -1) {
		p = strings.TrimSpace(strings.Trim(p, "#-*> "))
		if n := len(strings.Fields(p)); n >= 3 && n <= 60 {
			out = append(out, p)
		}
	}
	return out
}

// Extraction is the best sentence-level answer found in the retrieved hits.
type Extraction struct {
	Answer string
	Source string
	Score  float32
}

// Extract returns the retrieved sentence most similar to the query (and its source/score). The caller
// decides, via a confidence threshold, whether to return it directly (fast path) or fall back to the LLM.
func Extract(ctx context.Context, emb *Embedder, hits []Hit, query string) (Extraction, error) {
	type cand struct{ text, src string }
	var cands []cand
	for _, h := range hits {
		for _, s := range SplitSentences(h.Text) {
			cands = append(cands, cand{s, h.Source})
		}
	}
	if len(cands) == 0 {
		return Extraction{}, nil
	}
	texts := make([]string, 0, len(cands)+1)
	texts = append(texts, query)
	for _, c := range cands {
		texts = append(texts, c.text)
	}
	vecs, err := emb.Embed(ctx, texts)
	if err != nil {
		return Extraction{}, err
	}
	q := vecs[0]
	best, bestScore := 0, float32(-1)
	for i := 1; i < len(vecs); i++ {
		if s := cosine(q, vecs[i]); s > bestScore {
			bestScore, best = s, i-1
		}
	}
	return Extraction{Answer: cands[best].text, Source: cands[best].src, Score: bestScore}, nil
}

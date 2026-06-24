package library

import "testing"

func TestSplitSentences(t *testing.T) {
	txt := "The mascot is a heron. Memory must not exceed 1.84 GB; exactly 3 shards may be cached."
	s := SplitSentences(txt)
	if len(s) < 3 {
		t.Fatalf("expected >=3 candidate spans, got %d: %v", len(s), s)
	}
	for _, x := range s {
		if len(x) == 0 {
			t.Fatal("empty span")
		}
	}
}

// TestExtractGate checks the relevance gate via hand-made unit vectors: the query is closest to the
// "answer" sentence's vector (high score → fast path) and far from an unrelated one.
func TestExtractRanking(t *testing.T) {
	// cosine over unit vectors; the query vector must select the most-similar candidate.
	q := []float32{0.1, 0.95, 0.0}
	normalize(q)
	a := []float32{0.0, 1.0, 0.0} // "answer" — closest
	b := []float32{1.0, 0.0, 0.0} // unrelated
	if cosine(q, a) <= cosine(q, b) {
		t.Fatalf("expected answer vector to win: a=%.3f b=%.3f", cosine(q, a), cosine(q, b))
	}
}

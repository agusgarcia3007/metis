package library

import (
	"path/filepath"
	"testing"
)

func TestChunkText(t *testing.T) {
	words := ""
	for i := 0; i < 300; i++ {
		words += "w "
	}
	cs := ChunkText(words, "doc.md", 120, 30)
	if len(cs) < 3 {
		t.Fatalf("expected several chunks, got %d", len(cs))
	}
	for i, c := range cs {
		if c.Source != "doc.md" || c.Idx != i {
			t.Fatalf("chunk %d mislabeled: %+v", i, c)
		}
	}
}

func TestSearchAndPersist(t *testing.T) {
	// hand-made unit vectors: query is closest to chunk B.
	st := &Store{Model: "test", Dim: 3, Chunks: []Chunk{
		{Text: "A", Source: "a", Vec: []float32{1, 0, 0}},
		{Text: "B", Source: "b", Vec: []float32{0, 1, 0}},
		{Text: "C", Source: "c", Vec: []float32{0, 0, 1}},
	}}
	q := []float32{0.1, 0.9, 0.1}
	normalize(q)
	hits := st.Search(q, 2)
	if len(hits) != 2 || hits[0].Text != "B" {
		t.Fatalf("expected B first, got %+v", hits)
	}
	if hits[0].Score <= hits[1].Score {
		t.Fatalf("hits not sorted by score: %+v", hits)
	}

	path := filepath.Join(t.TempDir(), "idx.gob")
	if err := st.Save(path); err != nil {
		t.Fatal(err)
	}
	got, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Chunks) != 3 || got.Dim != 3 || got.Model != "test" {
		t.Fatalf("roundtrip mismatch: %+v", got)
	}
}

func TestCosineNormalized(t *testing.T) {
	a := []float32{3, 4}
	normalize(a)
	if d := cosine(a, a); d < 0.999 || d > 1.001 {
		t.Fatalf("self-cosine of unit vector should be 1, got %f", d)
	}
}

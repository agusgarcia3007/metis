package nano

import (
	"encoding/gob"
	"fmt"
	"io"
	"os"
)

// Save writes the model (config + all parameters) to path.
func (g *GPT) Save(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return g.Encode(f)
}

// Encode serializes config + parameters to w.
func (g *GPT) Encode(w io.Writer) error {
	enc := gob.NewEncoder(w)
	if err := enc.Encode(g.Cfg); err != nil {
		return err
	}
	for _, p := range g.Params() {
		if err := enc.Encode(p.Data); err != nil {
			return err
		}
	}
	return nil
}

// LoadGPT reconstructs a model previously written by Save.
func LoadGPT(path string) (*GPT, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	dec := gob.NewDecoder(f)
	var cfg Config
	if err := dec.Decode(&cfg); err != nil {
		return nil, err
	}
	g := NewGPT(cfg, 0)
	for _, p := range g.Params() {
		var data []float32
		if err := dec.Decode(&data); err != nil {
			return nil, err
		}
		if len(data) != len(p.Data) {
			return nil, fmt.Errorf("param size mismatch: got %d want %d", len(data), len(p.Data))
		}
		copy(p.Data, data)
	}
	return g, nil
}

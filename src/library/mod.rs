//! library is Metis's knowledge plane: the "Library" of the CLH-C architecture. It turns a corpus
//! of documents into a disk-resident, swappable index that a small Cortex retrieves from at query
//! time — knowledge-as-data: move facts OUT of the weights into data on disk.

mod extractive;
pub use extractive::{extract, split_sentences, Extraction};

use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{self, BufReader, BufWriter};
use std::time::Duration;

/// Chunk is one retrievable unit of knowledge with its embedding.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Chunk {
    pub text: String,
    pub source: String,
    pub idx: usize,
    #[serde(default)]
    pub vec: Vec<f32>,
}

/// Store is the on-disk knowledge index (swap the file, swap the brain's facts).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Store {
    pub model: String,
    pub dim: usize,
    pub chunks: Vec<Chunk>,
}

/// Hit is a retrieved chunk with its similarity score.
#[derive(Clone, Debug)]
pub struct Hit {
    pub chunk: Chunk,
    pub score: f32,
}

/// Embedder calls a local ollama embedding model.
pub struct Embedder {
    pub model: String,
    host: String,
    agent: ureq::Agent,
}

impl Embedder {
    /// NewEmbedder returns an embedder backed by the given ollama model (e.g. "all-minilm").
    pub fn new(model: &str, host: &str) -> Embedder {
        let host = if host.is_empty() {
            "http://127.0.0.1:11434".to_string()
        } else {
            host.to_string()
        };
        let agent = ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(120))
            .build();
        Embedder {
            model: model.to_string(),
            host,
            agent,
        }
    }

    /// Embed returns one vector per input text.
    pub fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        let body = serde_json::json!({ "model": self.model, "input": texts });
        let resp = self
            .agent
            .post(&format!("{}/api/embed", self.host))
            .set("Content-Type", "application/json")
            .send_json(body);
        let resp = match resp {
            Ok(r) => r,
            Err(ureq::Error::Status(code, _)) => {
                return Err(format!(
                    "embed: ollama {code} (is `ollama pull {}` done?)",
                    self.model
                ))
            }
            Err(e) => return Err(format!("embed: {e}")),
        };
        #[derive(Deserialize)]
        struct Out {
            embeddings: Vec<Vec<f32>>,
        }
        let mut out: Out = resp.into_json().map_err(|e| e.to_string())?;
        if out.embeddings.len() != texts.len() {
            return Err(format!(
                "embed: got {} vectors for {} texts",
                out.embeddings.len(),
                texts.len()
            ));
        }
        for v in out.embeddings.iter_mut() {
            normalize(v);
        }
        Ok(out.embeddings)
    }
}

fn normalize(v: &mut [f32]) {
    let mut n = 0.0f64;
    for &x in v.iter() {
        n += x as f64 * x as f64;
    }
    n = n.sqrt();
    if n == 0.0 {
        return;
    }
    let inv = (1.0 / n) as f32;
    for x in v.iter_mut() {
        *x *= inv;
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    // vectors are unit-normalized at ingest/query, so dot product == cosine
    let mut s = 0.0f32;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

/// ChunkText splits text into ~size-word windows with overlap, tagged with the source.
pub fn chunk_text(text: &str, source: &str, mut size: usize, mut overlap: usize) -> Vec<Chunk> {
    let fields: Vec<&str> = text.split_whitespace().collect();
    if fields.is_empty() {
        return Vec::new();
    }
    if size == 0 {
        size = 120;
    }
    if overlap >= size {
        overlap = size / 4;
    }
    let mut chunks = Vec::new();
    let mut start = 0usize;
    let mut idx = 0usize;
    while start < fields.len() {
        let end = (start + size).min(fields.len());
        chunks.push(Chunk {
            text: fields[start..end].join(" "),
            source: source.to_string(),
            idx,
            vec: Vec::new(),
        });
        idx += 1;
        if end == fields.len() {
            break;
        }
        start += size - overlap;
    }
    chunks
}

/// Build embeds the given chunks (in batches) into a Store.
pub fn build(emb: &Embedder, chunks: Vec<Chunk>) -> Result<Store, String> {
    const BATCH: usize = 32;
    let mut st = Store {
        model: emb.model.clone(),
        dim: 0,
        chunks: Vec::new(),
    };
    let mut i = 0;
    while i < chunks.len() {
        let j = (i + BATCH).min(chunks.len());
        let texts: Vec<String> = chunks[i..j].iter().map(|c| c.text.clone()).collect();
        let vecs = emb.embed(&texts)?;
        for (k, v) in vecs.into_iter().enumerate() {
            let mut c = chunks[i + k].clone();
            c.vec = v;
            st.chunks.push(c);
        }
        i = j;
    }
    if !st.chunks.is_empty() {
        st.dim = st.chunks[0].vec.len();
    }
    Ok(st)
}

impl Store {
    /// Search returns the top-k chunks most similar to the query embedding.
    pub fn search(&self, q_vec: &[f32], k: usize) -> Vec<Hit> {
        let mut hits: Vec<Hit> = self
            .chunks
            .iter()
            .map(|c| Hit {
                chunk: c.clone(),
                score: cosine(q_vec, &c.vec),
            })
            .collect();
        hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        if k < hits.len() {
            hits.truncate(k);
        }
        hits
    }

    /// Save persists the index to disk (this file IS the swappable knowledge).
    pub fn save(&self, path: &str) -> io::Result<()> {
        let w = BufWriter::new(File::create(path)?);
        bincode::serialize_into(w, self).map_err(io::Error::other)
    }
}

/// Load reconstructs a Store previously written by Save.
pub fn load(path: &str) -> io::Result<Store> {
    let r = BufReader::new(File::open(path)?);
    bincode::deserialize_from(r).map_err(io::Error::other)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_text_labels() {
        let words = "w ".repeat(300);
        let cs = chunk_text(&words, "doc.md", 120, 30);
        assert!(cs.len() >= 3, "expected several chunks, got {}", cs.len());
        for (i, c) in cs.iter().enumerate() {
            assert!(c.source == "doc.md" && c.idx == i, "chunk {i} mislabeled: {c:?}");
        }
    }

    #[test]
    fn search_and_persist() {
        let st = Store {
            model: "test".to_string(),
            dim: 3,
            chunks: vec![
                Chunk { text: "A".into(), source: "a".into(), idx: 0, vec: vec![1.0, 0.0, 0.0] },
                Chunk { text: "B".into(), source: "b".into(), idx: 0, vec: vec![0.0, 1.0, 0.0] },
                Chunk { text: "C".into(), source: "c".into(), idx: 0, vec: vec![0.0, 0.0, 1.0] },
            ],
        };
        let mut q = vec![0.1f32, 0.9, 0.1];
        normalize(&mut q);
        let hits = st.search(&q, 2);
        assert!(hits.len() == 2 && hits[0].chunk.text == "B", "expected B first, got {hits:?}");
        assert!(hits[0].score > hits[1].score, "hits not sorted by score");

        let dir = std::env::temp_dir();
        let path = dir.join(format!("metis0_idx_{}.bin", std::process::id()));
        let path = path.to_str().unwrap();
        st.save(path).unwrap();
        let got = load(path).unwrap();
        let _ = std::fs::remove_file(path);
        assert!(
            got.chunks.len() == 3 && got.dim == 3 && got.model == "test",
            "roundtrip mismatch: {got:?}"
        );
    }

    #[test]
    fn cosine_normalized() {
        let mut a = vec![3.0f32, 4.0];
        normalize(&mut a);
        let d = cosine(&a, &a);
        assert!((0.999..=1.001).contains(&d), "self-cosine should be 1, got {d}");
    }
}

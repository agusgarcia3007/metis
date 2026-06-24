//! Model persistence: write config + all parameters to disk and reload them.
//!
//! The Go original used `encoding/gob`; this port uses `bincode` (the closest Rust analog) for a
//! compact binary roundtrip with identical semantics. File paths are kept identical for CLI parity.

use super::model::{Config, Gpt};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{self, BufReader, BufWriter};

#[derive(Serialize, Deserialize)]
struct Saved {
    cfg: Config,
    params: Vec<Vec<f32>>,
}

fn enc_err(e: bincode::Error) -> io::Error {
    io::Error::other(e)
}

impl Gpt {
    /// Save writes the model (config + all parameters) to path.
    pub fn save(&self, path: &str) -> io::Result<()> {
        let params: Vec<Vec<f32>> = self.params().iter().map(|p| p.borrow().data.clone()).collect();
        let saved = Saved {
            cfg: self.cfg,
            params,
        };
        let w = BufWriter::new(File::create(path)?);
        bincode::serialize_into(w, &saved).map_err(enc_err)
    }
}

/// LoadGPT reconstructs a model previously written by Save.
pub fn load_gpt(path: &str) -> io::Result<Gpt> {
    let r = BufReader::new(File::open(path)?);
    let saved: Saved = bincode::deserialize_from(r).map_err(enc_err)?;
    let g = Gpt::new(saved.cfg, 0);
    let ps = g.params();
    if ps.len() != saved.params.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "param count mismatch: got {} want {}",
                saved.params.len(),
                ps.len()
            ),
        ));
    }
    for (p, data) in ps.iter().zip(saved.params.into_iter()) {
        let mut pb = p.borrow_mut();
        if pb.data.len() != data.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("param size mismatch: got {} want {}", data.len(), pb.data.len()),
            ));
        }
        pb.data.copy_from_slice(&data);
    }
    Ok(g)
}

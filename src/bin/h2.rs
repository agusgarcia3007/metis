//! H2 — measure false-SUPPORTED rate with the deterministic code verifier.
//!
//! The harness consumes candidate patches generated ahead of time (by the 0.6B Cortex, OpenCode,
//! or a fixed smoke set), so candidate generation is held constant while the verifier is tested.
//!
//! Build the pinned, offline sandbox:
//!   docker build -t metis-code-sandbox:phase5 sandbox/code
//!
//! Run the deterministic smoke set:
//!   cargo run --release --bin h2 -- bench/h2-smoke/dataset.json bench/results-h2-smoke.json

use metis_0::hands::verify_exec::{ExecError, ExecReport, ExecRequest, ExecVerifier};
use metis_0::verifier::VerifierKind;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize)]
struct Dataset {
    name: String,
    candidate_source: String,
    items: Vec<Item>,
}

#[derive(Debug, Deserialize)]
struct Item {
    id: String,
    depth: u8,
    expected_supported: bool,
    project: PathBuf,
    patch: PathBuf,
    #[serde(default)]
    held_out_tests: Option<PathBuf>,
}

#[derive(Debug, Default, Serialize)]
struct Cell {
    n_pos: u32,
    n_neg: u32,
    tp: u32,
    tn: u32,
    fab: u32,
    over_abstain: u32,
    uncertain: u32,
}

impl Cell {
    fn record(&mut self, expected_supported: bool, verdict: CandidateVerdict) {
        if expected_supported {
            self.n_pos += 1;
            match verdict {
                CandidateVerdict::Supported => self.tp += 1,
                CandidateVerdict::Unsupported => self.over_abstain += 1,
                CandidateVerdict::Uncertain => {
                    self.over_abstain += 1;
                    self.uncertain += 1;
                }
            }
        } else {
            self.n_neg += 1;
            match verdict {
                CandidateVerdict::Supported => self.fab += 1,
                CandidateVerdict::Unsupported => self.tn += 1,
                CandidateVerdict::Uncertain => self.uncertain += 1,
            }
        }
    }

    fn metrics(&self) -> Metrics {
        let tpr = ratio(self.tp, self.n_pos);
        let tnr = ratio(self.tn, self.n_neg);
        Metrics {
            tpr,
            tnr,
            fab_rate: ratio(self.fab, self.n_neg),
            balanced_accuracy: (tpr + tnr) / 2.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum CandidateVerdict {
    Supported,
    Unsupported,
    Uncertain,
}

impl CandidateVerdict {
    fn as_str(self) -> &'static str {
        match self {
            Self::Supported => "SUPPORTED",
            Self::Unsupported => "UNSUPPORTED",
            Self::Uncertain => "UNCERTAIN",
        }
    }
}

#[derive(Debug, Serialize)]
struct Metrics {
    tpr: f64,
    tnr: f64,
    fab_rate: f64,
    balanced_accuracy: f64,
}

#[derive(Debug, Serialize)]
struct CellReport {
    counts: Cell,
    metrics: Metrics,
}

#[derive(Debug, Serialize)]
struct ItemReport {
    id: String,
    depth: u8,
    expected_supported: bool,
    verdict: String,
    duration_ms: u128,
    execution: Option<ExecReport>,
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct Report {
    experiment: &'static str,
    dataset: String,
    candidate_source: String,
    sandbox_image: String,
    items: Vec<ItemReport>,
    by_depth: BTreeMap<String, CellReport>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("H2 failed: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let dataset_path = PathBuf::from(
        args.next()
            .unwrap_or_else(|| "bench/h2-smoke/dataset.json".to_string()),
    );
    let output_path = PathBuf::from(
        args.next()
            .unwrap_or_else(|| "bench/results-h2-smoke.json".to_string()),
    );
    if args.next().is_some() {
        return Err("usage: h2 [dataset.json] [results.json]".to_string());
    }

    let dataset_bytes = fs::read(&dataset_path)
        .map_err(|e| format!("read dataset {}: {e}", dataset_path.display()))?;
    let dataset: Dataset = serde_json::from_slice(&dataset_bytes)
        .map_err(|e| format!("parse dataset {}: {e}", dataset_path.display()))?;
    if dataset.items.is_empty() {
        return Err("dataset has no items".to_string());
    }

    let verifier = VerifierKind::Exec(Box::new(
        ExecVerifier::typescript().map_err(|e| format!("configure exec verifier: {e}"))?,
    ));
    let dataset_dir = dataset_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .canonicalize()
        .map_err(|e| format!("resolve dataset directory: {e}"))?;
    let sandbox_image = std::env::var("METIS_CODE_SANDBOX_IMAGE")
        .unwrap_or_else(|_| "metis-code-sandbox:phase5".to_string());

    eprintln!(
        "H2 dataset: {} ({} candidates, source: {})",
        dataset.name,
        dataset.items.len(),
        dataset.candidate_source
    );

    let total_items = dataset.items.len();
    let mut cells: BTreeMap<u8, Cell> = BTreeMap::new();
    let mut item_reports = Vec::with_capacity(total_items);
    for (index, item) in dataset.items.into_iter().enumerate() {
        let project_dir = dataset_dir.join(&item.project);
        let patch_path = dataset_dir.join(&item.patch);
        let patch = fs::read_to_string(&patch_path)
            .map_err(|e| format!("read patch {}: {e}", patch_path.display()))?;
        let held_out_tests = item
            .held_out_tests
            .as_ref()
            .map(|path| dataset_dir.join(path));
        let request = ExecRequest {
            project_dir,
            patch,
            held_out_tests,
        };
        let started = std::time::Instant::now();
        let (verdict, execution, error) = match verifier.verify_code(&request) {
            Ok(report) => {
                let verdict = if report.reward.fully_verified() {
                    CandidateVerdict::Supported
                } else {
                    CandidateVerdict::Unsupported
                };
                (verdict, Some(report), None)
            }
            // Policy rejection is a deterministic rejection of the candidate. Broken tools,
            // malformed tool output, and timeouts are uncertainty and never count as a true
            // negative.
            Err(ExecError::Policy(error)) => (
                CandidateVerdict::Unsupported,
                None,
                Some(format!("policy: {error}")),
            ),
            Err(error) => (CandidateVerdict::Uncertain, None, Some(error.to_string())),
        };
        let duration_ms = started.elapsed().as_millis();
        cells
            .entry(item.depth)
            .or_default()
            .record(item.expected_supported, verdict);
        eprintln!(
            "  [{}/{}] {} g{} expected={} -> {} ({} ms)",
            index + 1,
            total_items,
            item.id,
            item.depth,
            if item.expected_supported { "+" } else { "-" },
            verdict.as_str(),
            duration_ms
        );
        item_reports.push(ItemReport {
            id: item.id,
            depth: item.depth,
            expected_supported: item.expected_supported,
            verdict: verdict.as_str().to_string(),
            duration_ms,
            execution,
            error,
        });
    }

    let mut by_depth = BTreeMap::new();
    eprintln!("\nH2 deterministic verifier");
    eprintln!("g  n+  n-   TPR    TNR   fab%   bal-acc  uncertain");
    for (depth, cell) in cells {
        let metrics = cell.metrics();
        eprintln!(
            "g{depth} {:>3} {:>3}  {:.3}  {:.3}  {:>5.1}   {:.3}  {:>9}",
            cell.n_pos,
            cell.n_neg,
            metrics.tpr,
            metrics.tnr,
            metrics.fab_rate * 100.0,
            metrics.balanced_accuracy,
            cell.uncertain,
        );
        by_depth.insert(
            format!("g{depth}"),
            CellReport {
                counts: cell,
                metrics,
            },
        );
    }

    let report = Report {
        experiment: "H2",
        dataset: dataset.name,
        candidate_source: dataset.candidate_source,
        sandbox_image,
        items: item_reports,
        by_depth,
    };
    let serialized =
        serde_json::to_string_pretty(&report).map_err(|e| format!("serialize results: {e}"))?;
    fs::write(&output_path, format!("{serialized}\n"))
        .map_err(|e| format!("write {}: {e}", output_path.display()))?;
    println!("Raw data: {}", output_path.display());
    println!("SUCCESS CRITERION: fab% at g1 must be < 2% on the preregistered task set.");
    Ok(())
}

fn ratio(numerator: u32, denominator: u32) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

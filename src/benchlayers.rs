//! benchlayers — an offline, no-LLM benchmark of the two layers added from the Ornith-1.0 study:
//! the deterministic citation monitor (`crate::monitor`) and the adaptive scaffold router
//! (`crate::scaffold`).
//!
//! WHY THIS EXISTS (and what it is NOT). The headline Metis benchmark (`bench/benchmark.py`) and
//! the frontier coding benchmarks Ornith reports (SWE-Bench Verified, Terminal-Bench) all require a
//! *live model* — a running Cortex generating tokens. That is the right test for end-to-end answer
//! quality, but it cannot run in a sandbox with no model server. This harness measures the part of
//! the new work that is *deterministic by design*: given a candidate answer or a query, the monitor
//! and the router produce an exact, reproducible decision with zero inference. So we can run it
//! anywhere, on every commit, and get real numbers — `BASE` (the behaviour before this change) vs
//! `METIS` (with the layer enabled) — over a labelled adversarial suite.
//!
//! It does NOT claim anything about ARC-AGI / SWE-Bench scores, and it is not a substitute for the
//! live bare-vs-Metis run. See `bench/RESULTS.md` and `docs/design/07-…` for that honest accounting.

use crate::conductor::GvsConfig;
use crate::library::{Chunk, Hit};
use crate::monitor;
use crate::scaffold::Scaffold;

fn hit(source: &str) -> Hit {
    Hit { chunk: Chunk { text: "x".into(), source: source.into(), idx: 0, vec: vec![] }, score: 0.5 }
}

/// A labelled citation case: a candidate answer, how many sources were retrieved, and whether it
/// fabricates a citation (the ground-truth label).
struct CiteCase {
    answer: &'static str,
    n_sources: usize,
    fabricates: bool,
}

fn cite_suite() -> Vec<CiteCase> {
    vec![
        // ---- clean: every [n] references a real source (must be PRESERVED) ----
        CiteCase { answer: "The cap is 1.84 GB [1].", n_sources: 3, fabricates: false },
        CiteCase { answer: "A [1] and B [2].", n_sources: 2, fabricates: false },
        CiteCase { answer: "Stacked refs [1][2][3] all valid.", n_sources: 3, fabricates: false },
        CiteCase { answer: "Grouped [1, 2] citation.", n_sources: 2, fabricates: false },
        CiteCase { answer: "No citation at all, just prose.", n_sources: 3, fabricates: false },
        CiteCase { answer: "See [the docs](http://x); cite [1].", n_sources: 2, fabricates: false },
        CiteCase { answer: "Footnote [note] is not a citation; fact [1].", n_sources: 1, fabricates: false },
        // ---- fabricated: a [n] points past the evidence set (must be CAUGHT) ----
        CiteCase { answer: "The mascot is a heron [4].", n_sources: 3, fabricates: true },
        CiteCase { answer: "Memory caps at 1.84 GB [2], ratified 2034 [5].", n_sources: 3, fabricates: true },
        CiteCase { answer: "Grounded-looking claim [1].", n_sources: 0, fabricates: true },
        CiteCase { answer: "Off-by-one citation [3].", n_sources: 2, fabricates: true },
        CiteCase { answer: "Zero index is invalid [0].", n_sources: 2, fabricates: true },
        CiteCase { answer: "Mix of real [1] and invented [9].", n_sources: 4, fabricates: true },
    ]
}

/// A labelled routing case: a query, the shape of the retrieved evidence, and the expected scaffold.
struct RouteCase {
    query: &'static str,
    sources: Vec<&'static str>, // empty = no evidence; "http…" = web hit
    want: Scaffold,
}

fn route_suite() -> Vec<RouteCase> {
    vec![
        RouteCase { query: "hi there", sources: vec![], want: Scaffold::Direct },
        RouteCase { query: "What is 84937 * 2261?", sources: vec!["a.md"], want: Scaffold::Compute },
        RouteCase { query: "divide 100 by 7", sources: vec!["a.md"], want: Scaffold::Compute },
        RouteCase { query: "the product of 6 and 7 please", sources: vec!["a.md"], want: Scaffold::Compute },
        RouteCase { query: "who won the 2025 election", sources: vec!["https://news/x"], want: Scaffold::OpenDomain },
        RouteCase { query: "latest release notes", sources: vec!["zephyr.md", "https://site/y"], want: Scaffold::OpenDomain },
        RouteCase { query: "What is the protocol mascot?", sources: vec!["zephyr.md"], want: Scaffold::Factual },
        RouteCase { query: "In what year was version 4 ratified?", sources: vec!["std.md"], want: Scaffold::Factual },
        RouteCase { query: "Who chairs the Tessera working group?", sources: vec!["std.md", "wg.md"], want: Scaffold::Factual },
    ]
}

/// Report is the JSON-serialisable result of one run.
pub struct Report {
    pub cite_total: usize,
    pub cite_fabricated: usize,
    pub cite_clean: usize,
    /// BASE = no monitor: fabricated citations caught (always 0 — the prior loop never checked).
    pub base_caught: usize,
    pub base_false_rejects: usize,
    /// METIS = monitor on.
    pub metis_caught: usize,
    pub metis_false_rejects: usize,
    pub route_total: usize,
    pub route_correct: usize,
}

impl Report {
    pub fn to_json(&self) -> String {
        format!(
            "{{\n  \"citation_monitor\": {{\n    \"total\": {}, \"fabricated\": {}, \"clean\": {},\n    \"base\":  {{ \"caught\": {}, \"false_rejects\": {} }},\n    \"metis\": {{ \"caught\": {}, \"false_rejects\": {} }}\n  }},\n  \"scaffold_router\": {{ \"total\": {}, \"correct\": {} }}\n}}\n",
            self.cite_total, self.cite_fabricated, self.cite_clean,
            self.base_caught, self.base_false_rejects,
            self.metis_caught, self.metis_false_rejects,
            self.route_total, self.route_correct,
        )
    }
}

/// run executes both suites and returns the report (pure — no I/O).
pub fn run() -> Report {
    // --- Layer 1: citation monitor ---
    let cites = cite_suite();
    let cite_fabricated = cites.iter().filter(|c| c.fabricates).count();
    let cite_clean = cites.len() - cite_fabricated;
    let mut metis_caught = 0;
    let mut metis_false_rejects = 0;
    for c in &cites {
        let rejected = !monitor::check_citations(c.answer, c.n_sources).clean();
        if c.fabricates && rejected {
            metis_caught += 1;
        }
        if !c.fabricates && rejected {
            metis_false_rejects += 1;
        }
    }
    // BASE = the loop before this change: it had no Layer 1, so it caught 0 fabricated citations
    // and (by definition) never false-rejected a clean answer on citation grounds.
    let base_caught = 0;
    let base_false_rejects = 0;

    // --- Self-scaffolding: routing accuracy ---
    let routes = route_suite();
    let mut route_correct = 0;
    for r in &routes {
        let hits: Vec<Hit> = r.sources.iter().map(|s| hit(s)).collect();
        if Scaffold::select(r.query, &hits) == r.want {
            route_correct += 1;
        }
    }

    Report {
        cite_total: cites.len(),
        cite_fabricated,
        cite_clean,
        base_caught,
        base_false_rejects,
        metis_caught,
        metis_false_rejects,
        route_total: routes.len(),
        route_correct,
    }
}

/// run_and_print runs the suites, prints a markdown comparison (BASE vs METIS), and returns the
/// JSON so a caller can persist it. No model required.
pub fn run_and_print() -> String {
    let rep = run();

    println!("== Offline layer benchmark (no LLM) — BASE vs METIS ==\n");
    println!("Layer 1 — deterministic citation monitor");
    println!(
        "  suite: {} cases ({} fabricated-citation, {} clean)\n",
        rep.cite_total, rep.cite_fabricated, rep.cite_clean
    );
    println!("  | config | fabricated caught | clean preserved | false rejects |");
    println!("  |---|---:|---:|---:|");
    println!(
        "  | BASE (Layer 2 only) | {}/{} | {}/{} | {} |",
        rep.base_caught, rep.cite_fabricated, rep.cite_clean, rep.cite_clean, rep.base_false_rejects
    );
    println!(
        "  | METIS (Layer 1+2)   | {}/{} | {}/{} | {} |\n",
        rep.metis_caught,
        rep.cite_fabricated,
        rep.cite_clean - rep.metis_false_rejects,
        rep.cite_clean,
        rep.metis_false_rejects
    );

    println!("Self-scaffolding — per-query GVS routing");
    println!("  routing accuracy: {}/{} queries\n", rep.route_correct, rep.route_total);
    println!("  resulting GVS budget per scaffold (vs the old fixed config):");
    let base = GvsConfig::default();
    println!("  | scaffold | max_candidates | gen_temp | search_temp |");
    println!("  |---|---:|---:|---:|");
    println!(
        "  | (old fixed) | {} | {} | {} |",
        base.max_candidates, base.gen_temp, base.search_temp
    );
    for s in [Scaffold::Compute, Scaffold::OpenDomain, Scaffold::Factual, Scaffold::Direct] {
        let c = s.apply(GvsConfig::default());
        println!("  | {} | {} | {} | {} |", s.as_str(), c.max_candidates, c.gen_temp, c.search_temp);
    }
    println!();

    let json = rep.to_json();
    print!("{json}");
    json
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn monitor_catches_every_fabrication_and_keeps_every_clean() {
        let rep = run();
        // Perfect recall on fabricated citations, zero false rejects on clean answers.
        assert_eq!(rep.metis_caught, rep.cite_fabricated);
        assert_eq!(rep.metis_false_rejects, 0);
        // The BASE config (no Layer 1) catches none of them — that is the gap this layer closes.
        assert_eq!(rep.base_caught, 0);
    }

    #[test]
    fn scaffold_routing_is_perfect_on_labelled_suite() {
        let rep = run();
        assert_eq!(rep.route_correct, rep.route_total);
    }
}

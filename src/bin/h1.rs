//! h1 — the gating experiment for the "reasoning compiler" thesis.
//!
//! THE BET (docs/design/10): frontier-grade reasoning depth can be moved OUT of the weights and
//! into an external, verified search over ATOMIC steps — *if* the verify<generate asymmetry GROWS
//! as the step shrinks. A tiny model that can reliably VERIFY a 1-hop fact (even when it can't
//! reliably chain) can be composed by an external engine into a deep reasoner, because every step
//! is externally gated and error stops compounding.
//!
//! H1 isolates the single load-bearing claim: **does the verifier's accuracy decay with the number
//! of reasoning hops the claim requires?** We hold the EVIDENCE constant (the relevant relation
//! tables from the corpus) and vary ONLY the depth of the claim — 1, 2, or 3 hops — for both
//! supported and (plausibly) unsupported claims. We run the *exact production verifier*
//! (`VerifierKind::Llm`, the grounded LLM-judge) across model sizes.
//!
//! Prediction under the thesis: at g=1 (atomic) the small model is near-perfect; accuracy falls as
//! g grows. KILL CRITERION: if the 0.6B's balanced accuracy at g=1 is not ≥ ~0.9, the foundation is
//! gone — decomposing to atomic steps would not buy reliability, and the whole engine is moot.
//!
//! Run:  cargo run --release --bin h1 -- qwen3:0.6b qwen3:1.7b qwen3:4b
//! Out:  bench/results-h1.json  (+ a printed table)

use metis_0::kernel::OllamaKernel;
use metis_0::verifier::{Verdict, VerifierKind};
use std::fmt::Write as _;

/// One component's full relation row, drawn verbatim from bench/corpus/.
struct Comp {
    comp: &'static str,
    codename: &'static str,
    group: &'static str,
    chair: &'static str,
    budget: &'static str,
}

const COMPS: &[Comp] = &[
    Comp { comp: "Lumen",   codename: "Falconer",  group: "Tessera",     chair: "Dr. Ingrid Solvang",        budget: "512 MB" },
    Comp { comp: "Aster",   codename: "Pelican",   group: "Halyard",     chair: "Professor Amara Okonkwo",   budget: "192 MB" },
    Comp { comp: "Quill",   codename: "Sandpiper", group: "Marlinspike", chair: "Dr. Renzo Castellanos",     budget: "96 MB"  },
    Comp { comp: "Tideway", codename: "Curlew",    group: "Bellwether",  chair: "Dr. Yuki Tanabe",           budget: "64 MB"  },
];

/// Evidence for the CHAIR family: the three relation tables needed to chain
/// codename → component → working group → chair. Held CONSTANT across all chair items, so the only
/// thing that varies between a g=1 and a g=3 item is how many of these tables the verifier must
/// chain through. (Faithful to meridian-governance.md and meridian-spec.md.)
fn chair_evidence() -> String {
    let mut e = String::new();
    for c in COMPS {
        let _ = writeln!(e, "The {} component is codenamed {} in the reference build.", c.comp, c.codename);
    }
    for c in COMPS {
        let _ = writeln!(e, "The {} component is maintained by the {} working group.", c.comp, c.group);
    }
    for c in COMPS {
        let _ = writeln!(e, "The {} working group is chaired by {}.", c.group, c.chair);
    }
    e
}

/// Evidence for the BUDGET family: codename → component → memory budget. Held constant.
fn budget_evidence() -> String {
    let mut e = String::new();
    for c in COMPS {
        let _ = writeln!(e, "The {} component is codenamed {} in the reference build.", c.comp, c.codename);
    }
    for c in COMPS {
        let _ = writeln!(e, "{} has a memory budget of {}.", c.comp, c.budget);
    }
    e
}

#[derive(Clone)]
struct Item {
    #[allow(dead_code)] // kept for per-item provenance / future per-family breakdown
    family: &'static str,
    hops: u8,
    supported: bool,
    evidence: String,
    claim: String,
}

/// Build the full balanced dataset: for each component, supported + plausibly-wrong claims at each
/// granularity. Negatives take ANOTHER component's final answer (a value that *is* in the evidence,
/// just not the right one) — the realistic "fabrication" negative, not an obvious nonsense one.
fn dataset() -> Vec<Item> {
    let mut items = Vec::new();
    let chair_ev = chair_evidence();
    let budget_ev = budget_evidence();
    let n = COMPS.len();

    for (i, c) in COMPS.iter().enumerate() {
        let wrong = &COMPS[(i + 1) % n]; // a sibling: same evidence, wrong answer

        // ---- CHAIR family: g=1 (group→chair), g=2 (component→group→chair), g=3 (codename→…→chair)
        let chair_claims: [(u8, String); 3] = [
            (1, format!("The {} working group is chaired by {{X}}.", c.group)),
            (2, format!("The working group that maintains the {} component is chaired by {{X}}.", c.comp)),
            (3, format!("The working group that maintains the component codenamed {} is chaired by {{X}}.", c.codename)),
        ];
        for (hops, tmpl) in chair_claims {
            items.push(Item { family: "chair", hops, supported: true,
                evidence: chair_ev.clone(), claim: tmpl.replace("{X}", c.chair) });
            items.push(Item { family: "chair", hops, supported: false,
                evidence: chair_ev.clone(), claim: tmpl.replace("{X}", wrong.chair) });
        }

        // ---- BUDGET family: g=1 (component→budget), g=2 (codename→component→budget)
        let budget_claims: [(u8, String); 2] = [
            (1, format!("The {} component has a memory budget of {{X}}.", c.comp)),
            (2, format!("The component codenamed {} has a memory budget of {{X}}.", c.codename)),
        ];
        for (hops, tmpl) in budget_claims {
            items.push(Item { family: "budget", hops, supported: true,
                evidence: budget_ev.clone(), claim: tmpl.replace("{X}", c.budget) });
            items.push(Item { family: "budget", hops, supported: false,
                evidence: budget_ev.clone(), claim: tmpl.replace("{X}", wrong.budget) });
        }
    }
    items
}

/// Per-cell tally (one model × one granularity).
#[derive(Default)]
struct Cell {
    n_pos: u32,
    n_neg: u32,
    tp: u32,        // supported  → SUPPORTED   (correct)
    tn: u32,        // unsupported → UNSUPPORTED (correct)
    fab: u32,       // unsupported → SUPPORTED   (FABRICATION — the dangerous error)
    over_abst: u32, // supported   → UNSUPPORTED/UNCERTAIN (over-abstention)
    pos_unc: u32,
    neg_unc: u32,
}

impl Cell {
    fn tpr(&self) -> f64 { if self.n_pos == 0 { 0.0 } else { self.tp as f64 / self.n_pos as f64 } }
    fn tnr(&self) -> f64 { if self.n_neg == 0 { 0.0 } else { self.tn as f64 / self.n_neg as f64 } }
    fn fab_rate(&self) -> f64 { if self.n_neg == 0 { 0.0 } else { self.fab as f64 / self.n_neg as f64 } }
    fn balanced_acc(&self) -> f64 { (self.tpr() + self.tnr()) / 2.0 }
}

fn run_model(model: &str, items: &[Item]) -> std::collections::BTreeMap<u8, Cell> {
    let k = OllamaKernel::new(model, "");
    let v = VerifierKind::Llm;
    let mut cells: std::collections::BTreeMap<u8, Cell> = std::collections::BTreeMap::new();
    let total = items.len();
    for (idx, it) in items.iter().enumerate() {
        let verdict = v.verify(&k, &it.claim, &it.evidence);
        let cell = cells.entry(it.hops).or_default();
        if it.supported {
            cell.n_pos += 1;
            match verdict {
                Verdict::Supported => cell.tp += 1,
                Verdict::Uncertain => { cell.over_abst += 1; cell.pos_unc += 1; }
                Verdict::Unsupported => cell.over_abst += 1,
            }
        } else {
            cell.n_neg += 1;
            match verdict {
                Verdict::Unsupported => cell.tn += 1,
                Verdict::Supported => cell.fab += 1,
                Verdict::Uncertain => cell.neg_unc += 1,
            }
        }
        eprint!("\r  [{model}] {}/{total} (g{} {}) -> {:?}        ",
            idx + 1, it.hops, if it.supported { "+" } else { "-" }, verdict);
    }
    eprintln!();
    cells
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let models: Vec<String> = if args.is_empty() {
        vec!["qwen3:0.6b".into(), "qwen3:1.7b".into(), "qwen3:4b".into()]
    } else {
        args
    };

    let items = dataset();
    eprintln!("H1 dataset: {} items ({} models)", items.len(), models.len());

    // Quick reachability check.
    let probe = OllamaKernel::new(&models[0], "");
    if !probe.available() {
        eprintln!("ERROR: ollama not reachable at 127.0.0.1:11434 — start it first.");
        std::process::exit(1);
    }

    let mut json = String::from("{\n");
    let mut table = String::new();
    table.push_str("\n================  H1: verifier accuracy vs reasoning depth  ================\n");
    table.push_str("model            g  n+  n-   TPR    TNR   fab%   bal-acc\n");
    table.push_str("---------------------------------------------------------------\n");

    for (mi, model) in models.iter().enumerate() {
        eprintln!("\n>>> {model}");
        let cells = run_model(model, &items);
        let _ = writeln!(json, "  \"{model}\": {{");
        let n_gran = cells.len();
        for (gi, (g, c)) in cells.iter().enumerate() {
            let _ = writeln!(table,
                "{model:<15} g{g}  {:>2}  {:>2}  {:.3}  {:.3}  {:>4.0}   {:.3}",
                c.n_pos, c.n_neg, c.tpr(), c.tnr(), c.fab_rate() * 100.0, c.balanced_acc());
            let _ = write!(json,
                "    \"g{g}\": {{ \"n_pos\": {}, \"n_neg\": {}, \"tp\": {}, \"tn\": {}, \"fab\": {}, \"over_abstain\": {}, \"pos_uncertain\": {}, \"neg_uncertain\": {}, \"tpr\": {:.4}, \"tnr\": {:.4}, \"fab_rate\": {:.4}, \"balanced_acc\": {:.4} }}{}\n",
                c.n_pos, c.n_neg, c.tp, c.tn, c.fab, c.over_abst, c.pos_unc, c.neg_unc,
                c.tpr(), c.tnr(), c.fab_rate(), c.balanced_acc(),
                if gi + 1 < n_gran { "," } else { "" });
        }
        let _ = write!(json, "  }}{}\n", if mi + 1 < models.len() { "," } else { "" });
        table.push_str("---------------------------------------------------------------\n");
    }
    json.push_str("}\n");

    std::fs::write("bench/results-h1.json", &json).expect("write results");
    print!("{table}");
    println!("\nRaw data: bench/results-h1.json");
    println!("KILL CRITERION: 0.6B balanced-acc at g1 must be >= ~0.90 for the thesis to survive.");
}

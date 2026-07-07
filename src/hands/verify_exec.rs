//! Deterministic code verification backed by the isolated sandbox.
//!
//! A candidate is a unified diff. Before any code runs, the policy rejects test/config changes and
//! common skip/type-suppression primitives. The surviving patch is checked by syntax, type, lint,
//! and test gates. Infrastructure failures are returned as errors so callers abstain rather than
//! silently treating a broken verifier as a rejected candidate.

use crate::hands::sandbox::{Sandbox, SandboxCommand, SandboxConfig, SandboxError, SandboxOutput};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::{Component, Path, PathBuf};
use std::time::Instant;

const MAX_PATCH_BYTES: usize = 2 * 1024 * 1024;
const MAX_TOUCHED_FILES: usize = 256;

/// The exact verifier signal consumed by code search.
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
pub struct Reward {
    pub compiles: bool,
    pub typechecks: bool,
    pub lint_clean: bool,
    /// True only when the test runner itself exited successfully and reported success.
    pub tests_successful: bool,
    pub tests_passed: u32,
    pub tests_total: u32,
}

impl Reward {
    pub fn tests_full(&self) -> bool {
        self.tests_successful && self.tests_total > 0 && self.tests_passed == self.tests_total
    }

    pub fn fully_verified(&self) -> bool {
        self.compiles && self.typechecks && self.lint_clean && self.tests_full()
    }

    /// Dense score for search. Syntax and types unlock the later rewards; tests then contribute
    /// proportionally, with a separate bonus for a complete pass.
    pub fn score(&self) -> f64 {
        if !self.compiles {
            return 0.0;
        }
        let mut score = 1.0;
        if !self.typechecks {
            return score;
        }
        score += 1.0;
        if self.lint_clean {
            score += 0.5;
        }
        if self.tests_total > 0 {
            score += 2.0 * self.tests_passed as f64 / self.tests_total as f64;
            if self.tests_full() {
                score += 1.0;
            }
        }
        score
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct GateResult {
    pub success: bool,
    pub exit_code: i32,
    pub duration_ms: u128,
    pub stdout: String,
    pub stderr: String,
    pub output_truncated: bool,
}

impl From<SandboxOutput> for GateResult {
    fn from(output: SandboxOutput) -> Self {
        Self {
            success: output.exit_code == 0,
            exit_code: output.exit_code,
            duration_ms: output.duration.as_millis(),
            stdout: output.stdout,
            stderr: output.stderr,
            output_truncated: output.output_truncated,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TestResult {
    pub gate: GateResult,
    pub passed: u32,
    pub total: u32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ExecReport {
    pub reward: Reward,
    pub compile: GateResult,
    pub typecheck: GateResult,
    pub lint: GateResult,
    pub tests: TestResult,
    pub duration_ms: u128,
}

#[derive(Clone, Debug)]
pub struct ExecRequest {
    pub project_dir: PathBuf,
    pub patch: String,
    /// Files copied into the sandbox only after the candidate patch is applied.
    pub held_out_tests: Option<PathBuf>,
}

#[derive(Clone, Debug)]
pub struct PatchPolicy {
    /// Additional exact paths or directory prefixes protected from candidate edits.
    pub protected_paths: Vec<String>,
}

impl Default for PatchPolicy {
    fn default() -> Self {
        Self {
            protected_paths: vec![
                ".github".into(),
                "package.json".into(),
                "package-lock.json".into(),
                "pnpm-lock.yaml".into(),
                "yarn.lock".into(),
                "bun.lock".into(),
                "bun.lockb".into(),
                "tsconfig.json".into(),
                "vitest.config.ts".into(),
                "vitest.config.js".into(),
                "jest.config.ts".into(),
                "jest.config.js".into(),
                "eslint.config.js".into(),
                "eslint.config.mjs".into(),
                ".eslintrc".into(),
                "metis-code-agent.json".into(),
            ],
        }
    }
}

#[derive(Clone, Debug)]
pub struct ExecConfig {
    pub compile: SandboxCommand,
    pub typecheck: SandboxCommand,
    pub lint: SandboxCommand,
    pub tests: SandboxCommand,
    pub policy: PatchPolicy,
}

impl Default for ExecConfig {
    fn default() -> Self {
        Self {
            compile: SandboxCommand::new("metis-ts-parse", std::iter::empty::<String>()),
            typecheck: SandboxCommand::new("metis-ts-typecheck", std::iter::empty::<String>()),
            lint: SandboxCommand::new("metis-ts-lint", std::iter::empty::<String>()),
            tests: SandboxCommand::new("metis-vitest", ["run", "--reporter=json"]),
            policy: PatchPolicy::default(),
        }
    }
}

#[derive(Debug)]
pub enum ExecError {
    Policy(String),
    Sandbox(SandboxError),
    InvalidTestReport(String),
    WrongVerifierKind,
}

impl fmt::Display for ExecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Policy(msg) => write!(f, "candidate patch rejected by policy: {msg}"),
            Self::Sandbox(err) => write!(f, "{err}"),
            Self::InvalidTestReport(msg) => write!(f, "invalid test report: {msg}"),
            Self::WrongVerifierKind => write!(f, "verifier is not configured for code execution"),
        }
    }
}

impl std::error::Error for ExecError {}

impl From<SandboxError> for ExecError {
    fn from(value: SandboxError) -> Self {
        Self::Sandbox(value)
    }
}

/// TypeScript pilot verifier from Phase 5.0.
#[derive(Clone, Debug)]
pub struct ExecVerifier {
    sandbox: Sandbox,
    config: ExecConfig,
}

impl ExecVerifier {
    pub fn new(sandbox: SandboxConfig, config: ExecConfig) -> Result<Self, ExecError> {
        Ok(Self {
            sandbox: Sandbox::new(sandbox)?,
            config,
        })
    }

    pub fn typescript() -> Result<Self, ExecError> {
        Self::new(SandboxConfig::default(), ExecConfig::default())
    }

    pub fn verify(&self, request: &ExecRequest) -> Result<ExecReport, ExecError> {
        validate_patch(&request.patch, &self.config.policy)?;
        let started = Instant::now();

        let compile: GateResult = self
            .sandbox
            .run(&request.project_dir, &request.patch, &self.config.compile)?
            .into();
        let typecheck: GateResult = self
            .sandbox
            .run(&request.project_dir, &request.patch, &self.config.typecheck)?
            .into();
        let lint: GateResult = self
            .sandbox
            .run(&request.project_dir, &request.patch, &self.config.lint)?
            .into();
        let tests_output = self.sandbox.run_with_held_out(
            &request.project_dir,
            &request.patch,
            request.held_out_tests.as_deref(),
            &self.config.tests,
        )?;
        let tests_gate: GateResult = tests_output.into();
        let (reported_success, tests_passed, tests_total) =
            parse_vitest_counts(&tests_gate.stdout)?;
        let tests_successful = tests_gate.success && reported_success;
        let tests = TestResult {
            gate: tests_gate,
            passed: tests_passed,
            total: tests_total,
        };

        let reward = Reward {
            compiles: compile.success,
            typechecks: typecheck.success,
            lint_clean: lint.success,
            tests_successful,
            tests_passed,
            tests_total,
        };

        Ok(ExecReport {
            reward,
            compile,
            typecheck,
            lint,
            tests,
            duration_ms: started.elapsed().as_millis(),
        })
    }
}

/// Reject patch surfaces that would let a policy weaken or bypass the oracle.
pub fn validate_patch(patch: &str, policy: &PatchPolicy) -> Result<(), ExecError> {
    if patch.trim().is_empty() {
        return Err(ExecError::Policy("patch is empty".to_string()));
    }
    if patch.len() > MAX_PATCH_BYTES {
        return Err(ExecError::Policy(format!(
            "patch exceeds {MAX_PATCH_BYTES} bytes"
        )));
    }

    let mut touched = Vec::new();
    for line in patch.lines() {
        if let Some(paths) = line.strip_prefix("diff --git ") {
            let paths: Vec<_> = paths.split_whitespace().collect();
            if paths.len() != 2 {
                return Err(ExecError::Policy(
                    "quoted or malformed diff paths are unsupported".to_string(),
                ));
            }
            for raw_path in paths {
                touched.push(normalize_patch_path(raw_path)?);
            }
        }

        if line.starts_with('+') && !line.starts_with("+++") {
            reject_forbidden_addition(&line[1..])?;
        }
    }
    touched.sort();
    touched.dedup();
    if touched.is_empty() {
        return Err(ExecError::Policy(
            "patch contains no diff --git file header".to_string(),
        ));
    }
    if touched.len() > MAX_TOUCHED_FILES {
        return Err(ExecError::Policy(format!(
            "patch touches more than {MAX_TOUCHED_FILES} files"
        )));
    }

    for path in touched {
        if is_test_path(&path) {
            return Err(ExecError::Policy(format!(
                "tests are immutable ({})",
                path.display()
            )));
        }
        if is_intrinsically_protected(&path)
            || is_tooling_config(&path)
            || is_configured_protected(&path, policy)
        {
            return Err(ExecError::Policy(format!(
                "protected path modified ({})",
                path.display()
            )));
        }
    }
    Ok(())
}

fn normalize_patch_path(raw: &str) -> Result<PathBuf, ExecError> {
    if raw.starts_with('"') {
        return Err(ExecError::Policy(
            "quoted patch paths are unsupported".to_string(),
        ));
    }
    let raw = raw
        .strip_prefix("a/")
        .or_else(|| raw.strip_prefix("b/"))
        .unwrap_or(raw);
    let path = Path::new(raw);
    if raw.is_empty()
        || path
            .components()
            .any(|part| !matches!(part, Component::Normal(_)))
    {
        return Err(ExecError::Policy(format!("unsafe patch path: {raw}")));
    }
    Ok(path.to_path_buf())
}

fn is_test_path(path: &Path) -> bool {
    let lower = path.to_string_lossy().to_ascii_lowercase();
    let file = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    lower.split('/').any(|part| {
        matches!(
            part,
            "test" | "tests" | "__test__" | "__tests__" | "spec" | "specs"
        )
    }) || file.contains(".test.")
        || file.contains(".spec.")
        || file.starts_with("test_")
        || file.ends_with("_test.ts")
        || file.ends_with("_test.tsx")
}

fn is_intrinsically_protected(path: &Path) -> bool {
    let lower = path.to_string_lossy().to_ascii_lowercase();
    lower == ".git"
        || lower.starts_with(".git/")
        || lower == "node_modules"
        || lower.starts_with("node_modules/")
        || lower == "dockerfile"
        || lower.starts_with("dockerfile.")
        || lower == "docker-compose.yml"
        || lower == "docker-compose.yaml"
}

fn is_tooling_config(path: &Path) -> bool {
    let file = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    matches!(
        file.as_str(),
        "package.json"
            | "package-lock.json"
            | "pnpm-lock.yaml"
            | "yarn.lock"
            | "bun.lock"
            | "bun.lockb"
            | ".npmrc"
            | "deno.json"
            | "deno.jsonc"
    ) || (file.starts_with("tsconfig") && file.ends_with(".json"))
        || file.starts_with("vitest.config.")
        || file.starts_with("vite.config.")
        || file.starts_with("jest.config.")
        || file.starts_with("eslint.config.")
        || file.starts_with(".eslintrc")
        || file == ".env"
        || file.starts_with(".env.")
}

fn is_configured_protected(path: &Path, policy: &PatchPolicy) -> bool {
    let rendered = path.to_string_lossy();
    policy.protected_paths.iter().any(|protected| {
        rendered == protected.as_str()
            || rendered
                .strip_prefix(protected)
                .is_some_and(|tail| tail.starts_with('/'))
    })
}

fn reject_forbidden_addition(line: &str) -> Result<(), ExecError> {
    let compact: String = line
        .to_ascii_lowercase()
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect();
    let forbidden = [
        ".skip(",
        ".only(",
        "test.todo(",
        "it.todo(",
        "describe.todo(",
        "xit(",
        "xdescribe(",
        "@ts-ignore",
        "@ts-expect-error",
        "eslint-disable",
        "istanbulignore",
        "c8ignore",
    ];
    if let Some(token) = forbidden.iter().find(|token| compact.contains(**token)) {
        return Err(ExecError::Policy(format!(
            "forbidden verification bypass added ({token})"
        )));
    }
    Ok(())
}

fn parse_vitest_counts(stdout: &str) -> Result<(bool, u32, u32), ExecError> {
    let trimmed = stdout.trim();
    let parsed = serde_json::from_str::<serde_json::Value>(trimmed).or_else(|_| {
        let start = trimmed.find('{').unwrap_or(trimmed.len());
        let end = trimmed.rfind('}').map(|index| index + 1).unwrap_or(0);
        if start < end {
            serde_json::from_str::<serde_json::Value>(&trimmed[start..end])
        } else {
            serde_json::from_str::<serde_json::Value>("")
        }
    });
    let value = parsed.map_err(|e| ExecError::InvalidTestReport(e.to_string()))?;
    let success = value
        .get("success")
        .and_then(|v| v.as_bool())
        .ok_or_else(|| {
            ExecError::InvalidTestReport("missing success in Vitest JSON".to_string())
        })?;
    let passed = value
        .get("numPassedTests")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| {
            ExecError::InvalidTestReport("missing numPassedTests in Vitest JSON".to_string())
        })?;
    let total = value
        .get("numTotalTests")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| {
            ExecError::InvalidTestReport("missing numTotalTests in Vitest JSON".to_string())
        })?;
    let passed = u32::try_from(passed)
        .map_err(|_| ExecError::InvalidTestReport("passed test count exceeds u32".to_string()))?;
    let total = u32::try_from(total)
        .map_err(|_| ExecError::InvalidTestReport("total test count exceeds u32".to_string()))?;
    Ok((success, passed, total))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAFE_PATCH: &str = r#"diff --git a/src/math.ts b/src/math.ts
--- a/src/math.ts
+++ b/src/math.ts
@@ -1 +1 @@
-export const add = (a: number, b: number) => a + b;
+export const add = (a: number, b: number) => b + a;
"#;

    #[test]
    fn reward_is_dense_and_requires_nonempty_tests() {
        let partial = Reward {
            compiles: true,
            typechecks: true,
            lint_clean: true,
            tests_successful: false,
            tests_passed: 2,
            tests_total: 4,
        };
        let full = Reward {
            tests_successful: true,
            tests_passed: 4,
            ..partial.clone()
        };
        assert!(partial.score() > 2.5);
        assert!(full.score() > partial.score());
        assert!(!partial.fully_verified());
        assert!(full.fully_verified());
        assert!(!Reward {
            compiles: true,
            typechecks: true,
            lint_clean: true,
            tests_successful: true,
            tests_passed: 0,
            tests_total: 0,
        }
        .fully_verified());
    }

    #[test]
    fn patch_policy_accepts_source_only_change() {
        validate_patch(SAFE_PATCH, &PatchPolicy::default()).unwrap();
    }

    #[test]
    fn patch_policy_rejects_test_and_config_changes() {
        let test_patch = SAFE_PATCH.replace("src/math.ts", "tests/math.test.ts");
        assert!(validate_patch(&test_patch, &PatchPolicy::default()).is_err());

        let config_patch = SAFE_PATCH.replace("src/math.ts", "package.json");
        assert!(validate_patch(&config_patch, &PatchPolicy::default()).is_err());

        let nested_config = SAFE_PATCH.replace("src/math.ts", "packages/app/tsconfig.build.json");
        assert!(validate_patch(&nested_config, &PatchPolicy::default()).is_err());
    }

    #[test]
    fn patch_policy_rejects_skip_and_type_suppression() {
        let skipped = SAFE_PATCH.replace(
            "+export const add = (a: number, b: number) => b + a;",
            "+test.skip('oracle', () => {});",
        );
        assert!(validate_patch(&skipped, &PatchPolicy::default()).is_err());

        let ignored = SAFE_PATCH.replace(
            "+export const add = (a: number, b: number) => b + a;",
            "+// @ts-expect-error",
        );
        assert!(validate_patch(&ignored, &PatchPolicy::default()).is_err());
    }

    #[test]
    fn patch_policy_rejects_path_traversal() {
        let traversal = SAFE_PATCH.replace("src/math.ts", "../outside.ts");
        assert!(validate_patch(&traversal, &PatchPolicy::default()).is_err());
    }

    #[test]
    fn parses_vitest_json_counts() {
        let report = r#"{"numTotalTests":7,"numPassedTests":5,"success":false}"#;
        assert_eq!(parse_vitest_counts(report).unwrap(), (false, 5, 7));
    }
}

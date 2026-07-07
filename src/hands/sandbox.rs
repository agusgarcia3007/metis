//! Container-backed execution for untrusted code candidates.
//!
//! The host repository and candidate patch are mounted read-only. The sandbox image copies them
//! into an ephemeral tmpfs, applies the patch there, and executes one argv-only command. There is
//! no shell interpolation of model-controlled content.

use std::fmt;
use std::fs;
use std::io::{self, Read};
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

const OUTPUT_LIMIT_BYTES: usize = 1_048_576;
const SETUP_ERROR_MARKER: &str = "METIS_SANDBOX_SETUP_ERROR:";
static CONTAINER_SEQUENCE: AtomicU64 = AtomicU64::new(1);

/// Resource and isolation settings for one code-verification command.
#[derive(Clone, Debug)]
pub struct SandboxConfig {
    /// Docker-compatible runtime executable.
    pub runtime: String,
    /// Image implementing the `sandbox/code/entrypoint.sh` contract.
    pub image: String,
    pub cpus: f32,
    pub memory_mb: u64,
    pub workspace_mb: u64,
    pub pids_limit: u32,
    pub timeout: Duration,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            runtime: std::env::var("METIS_CONTAINER_RUNTIME")
                .unwrap_or_else(|_| "docker".to_string()),
            image: std::env::var("METIS_CODE_SANDBOX_IMAGE")
                .unwrap_or_else(|_| "metis-code-sandbox:phase5".to_string()),
            cpus: 1.0,
            memory_mb: 1_024,
            workspace_mb: 1_024,
            pids_limit: 256,
            timeout: Duration::from_secs(120),
        }
    }
}

/// A command passed directly as argv to the sandbox entrypoint.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SandboxCommand {
    pub program: String,
    pub args: Vec<String>,
}

impl SandboxCommand {
    pub fn new(
        program: impl Into<String>,
        args: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self {
            program: program.into(),
            args: args.into_iter().map(Into::into).collect(),
        }
    }
}

/// Captured result of a command which ran successfully at the container-infrastructure level.
#[derive(Clone, Debug)]
pub struct SandboxOutput {
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
    pub duration: Duration,
    pub output_truncated: bool,
}

#[derive(Debug)]
pub enum SandboxError {
    InvalidConfig(String),
    Io(String),
    Runtime(String),
    Setup(String),
    TimedOut { after: Duration },
}

impl fmt::Display for SandboxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "invalid sandbox config: {msg}"),
            Self::Io(msg) => write!(f, "sandbox I/O error: {msg}"),
            Self::Runtime(msg) => write!(f, "container runtime error: {msg}"),
            Self::Setup(msg) => write!(f, "sandbox setup error: {msg}"),
            Self::TimedOut { after } => {
                write!(
                    f,
                    "sandbox command timed out after {} ms",
                    after.as_millis()
                )
            }
        }
    }
}

impl std::error::Error for SandboxError {}

/// Runs candidate code in an ephemeral, networkless container.
#[derive(Clone, Debug)]
pub struct Sandbox {
    config: SandboxConfig,
}

impl Sandbox {
    pub fn new(config: SandboxConfig) -> Result<Self, SandboxError> {
        validate_config(&config)?;
        Ok(Self { config })
    }

    pub fn config(&self) -> &SandboxConfig {
        &self.config
    }

    /// Apply `patch` to a private tmpfs copy of `project_dir`, then run `command`.
    pub fn run(
        &self,
        project_dir: &Path,
        patch: &str,
        command: &SandboxCommand,
    ) -> Result<SandboxOutput, SandboxError> {
        self.run_with_held_out(project_dir, patch, None, command)
    }

    /// Like [`Self::run`], but injects verifier-only files after applying the candidate patch.
    /// This is used for held-out tests that must not be visible to the patch generator.
    pub fn run_with_held_out(
        &self,
        project_dir: &Path,
        patch: &str,
        held_out_dir: Option<&Path>,
        command: &SandboxCommand,
    ) -> Result<SandboxOutput, SandboxError> {
        if command.program.trim().is_empty() {
            return Err(SandboxError::InvalidConfig(
                "sandbox command program cannot be empty".to_string(),
            ));
        }

        let project_dir = project_dir.canonicalize().map_err(|e| {
            SandboxError::Io(format!("canonicalize {}: {e}", project_dir.display()))
        })?;
        if !project_dir.is_dir() {
            return Err(SandboxError::InvalidConfig(format!(
                "project path is not a directory: {}",
                project_dir.display()
            )));
        }
        reject_mount_unsafe_path(&project_dir)?;
        let held_out_dir = held_out_dir
            .map(|path| {
                let path = path.canonicalize().map_err(|e| {
                    SandboxError::Io(format!("canonicalize held-out {}: {e}", path.display()))
                })?;
                if !path.is_dir() {
                    return Err(SandboxError::InvalidConfig(format!(
                        "held-out path is not a directory: {}",
                        path.display()
                    )));
                }
                reject_mount_unsafe_path(&path)?;
                Ok(path)
            })
            .transpose()?;

        let patch_dir = tempfile::tempdir()
            .map_err(|e| SandboxError::Io(format!("create candidate tempdir: {e}")))?;
        let patch_path = patch_dir.path().join("candidate.patch");
        fs::write(&patch_path, patch)
            .map_err(|e| SandboxError::Io(format!("write candidate patch: {e}")))?;
        reject_mount_unsafe_path(&patch_path)?;

        let container_name = container_name();
        let mut runtime = Command::new(&self.config.runtime);
        runtime
            .arg("run")
            .arg("--name")
            .arg(&container_name)
            .arg("--rm")
            .arg("--network")
            .arg("none")
            .arg("--cpus")
            .arg(self.config.cpus.to_string())
            .arg("--memory")
            .arg(format!("{}m", self.config.memory_mb))
            .arg("--memory-swap")
            .arg(format!("{}m", self.config.memory_mb))
            .arg("--pids-limit")
            .arg(self.config.pids_limit.to_string())
            .arg("--read-only")
            .arg("--cap-drop")
            .arg("ALL")
            .arg("--security-opt")
            .arg("no-new-privileges")
            .arg("--cap-add")
            .arg("SETUID")
            .arg("--cap-add")
            .arg("SETGID")
            .arg("--tmpfs")
            .arg(format!(
                "/workspace:rw,exec,nosuid,nodev,size={}m",
                self.config.workspace_mb
            ))
            .arg("--tmpfs")
            .arg("/tmp:rw,exec,nosuid,nodev,size=256m,uid=1000,gid=1000")
            .arg("--mount")
            .arg(format!(
                "type=bind,src={},dst=/input,readonly",
                project_dir.display()
            ))
            .arg("--mount")
            .arg(format!(
                "type=bind,src={},dst=/candidate/candidate.patch,readonly",
                patch_path.display()
            ))
            .arg("--env")
            .arg("CI=1")
            .arg("--env")
            .arg("HOME=/tmp")
            .arg("--env")
            .arg("NO_COLOR=1")
            .arg("--workdir")
            .arg("/workspace");
        if let Some(held_out_dir) = held_out_dir {
            runtime.arg("--mount").arg(format!(
                "type=bind,src={},dst=/held-out,readonly",
                held_out_dir.display()
            ));
        }
        runtime
            .arg(&self.config.image)
            .arg(&command.program)
            .args(&command.args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let started = Instant::now();
        let mut child = runtime
            .spawn()
            .map_err(|e| SandboxError::Runtime(format!("start {}: {e}", self.config.runtime)))?;

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| SandboxError::Io("capture runtime stdout".to_string()))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| SandboxError::Io("capture runtime stderr".to_string()))?;
        let stdout_reader = thread::spawn(move || read_capped(stdout));
        let stderr_reader = thread::spawn(move || read_capped(stderr));

        let status = loop {
            match child.try_wait() {
                Ok(Some(status)) => break status,
                Ok(None) if started.elapsed() < self.config.timeout => {
                    thread::sleep(Duration::from_millis(10));
                }
                Ok(None) => {
                    cleanup_container(&self.config.runtime, &container_name);
                    let _ = child.kill();
                    let _ = child.wait();
                    let _ = stdout_reader.join();
                    let _ = stderr_reader.join();
                    return Err(SandboxError::TimedOut {
                        after: self.config.timeout,
                    });
                }
                Err(e) => {
                    cleanup_container(&self.config.runtime, &container_name);
                    return Err(SandboxError::Io(format!("wait for container runtime: {e}")));
                }
            }
        };

        let (stdout, stdout_truncated) = join_reader(stdout_reader, "stdout")?;
        let (stderr, stderr_truncated) = join_reader(stderr_reader, "stderr")?;
        let exit_code = status.code().unwrap_or(128);

        // Docker/Podman reserve 125 for runtime failures. 126/127 usually mean the configured
        // verifier executable cannot be invoked, which is infrastructure/config, not a bad patch.
        if matches!(exit_code, 125..=127) {
            return Err(SandboxError::Runtime(format!(
                "{} exited {exit_code}: {}",
                self.config.runtime,
                stderr.trim()
            )));
        }
        if let Some((_, setup_error)) = stderr.split_once(SETUP_ERROR_MARKER) {
            return Err(SandboxError::Setup(setup_error.trim().to_string()));
        }

        Ok(SandboxOutput {
            exit_code,
            stdout,
            stderr,
            duration: started.elapsed(),
            output_truncated: stdout_truncated || stderr_truncated,
        })
    }
}

fn validate_config(config: &SandboxConfig) -> Result<(), SandboxError> {
    if config.runtime.trim().is_empty() || config.image.trim().is_empty() {
        return Err(SandboxError::InvalidConfig(
            "runtime and image must be non-empty".to_string(),
        ));
    }
    if !config.cpus.is_finite() || config.cpus <= 0.0 {
        return Err(SandboxError::InvalidConfig(
            "cpus must be finite and greater than zero".to_string(),
        ));
    }
    if config.memory_mb < 64 || config.workspace_mb < 64 || config.pids_limit == 0 {
        return Err(SandboxError::InvalidConfig(
            "memory/workspace must be >=64 MB and pids_limit > 0".to_string(),
        ));
    }
    if config.timeout.is_zero() {
        return Err(SandboxError::InvalidConfig(
            "timeout must be greater than zero".to_string(),
        ));
    }
    Ok(())
}

fn reject_mount_unsafe_path(path: &Path) -> Result<(), SandboxError> {
    let rendered = path.to_string_lossy();
    if rendered.contains(',') {
        return Err(SandboxError::InvalidConfig(format!(
            "mount paths containing commas are unsupported: {rendered}"
        )));
    }
    Ok(())
}

fn container_name() -> String {
    let sequence = CONTAINER_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    format!("metis-code-{}-{sequence}", std::process::id())
}

fn cleanup_container(runtime: &str, name: &str) {
    let _ = Command::new(runtime)
        .args(["rm", "--force", name])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
}

fn read_capped(mut reader: impl Read) -> io::Result<(String, bool)> {
    let mut retained = Vec::new();
    let mut buf = [0_u8; 8_192];
    let mut truncated = false;
    loop {
        let read = reader.read(&mut buf)?;
        if read == 0 {
            break;
        }
        let remaining = OUTPUT_LIMIT_BYTES.saturating_sub(retained.len());
        let keep = remaining.min(read);
        retained.extend_from_slice(&buf[..keep]);
        truncated |= keep < read;
    }
    Ok((String::from_utf8_lossy(&retained).into_owned(), truncated))
}

fn join_reader(
    reader: thread::JoinHandle<io::Result<(String, bool)>>,
    stream: &str,
) -> Result<(String, bool), SandboxError> {
    reader
        .join()
        .map_err(|_| SandboxError::Io(format!("{stream} reader panicked")))?
        .map_err(|e| SandboxError::Io(format!("read runtime {stream}: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_resource_limits() {
        let config = SandboxConfig {
            cpus: 0.0,
            ..SandboxConfig::default()
        };
        assert!(Sandbox::new(config).is_err());
    }

    #[test]
    fn command_keeps_arguments_separate() {
        let command = SandboxCommand::new("printf", ["%s", "$(not-a-shell)"]);
        assert_eq!(command.program, "printf");
        assert_eq!(command.args, ["%s", "$(not-a-shell)"]);
    }

    #[test]
    fn capped_reader_drains_but_bounds_retained_output() {
        let input = vec![b'x'; OUTPUT_LIMIT_BYTES + 100];
        let (output, truncated) = read_capped(input.as_slice()).unwrap();
        assert_eq!(output.len(), OUTPUT_LIMIT_BYTES);
        assert!(truncated);
    }
}

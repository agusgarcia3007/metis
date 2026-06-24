//! The clock tool: live local date/time the frozen weights of a small model cannot know.

use chrono::Local;

/// Now returns the current local date and time.
pub fn now() -> Result<String, String> {
    Ok(Local::now().format("%A, %Y-%m-%d %H:%M %Z").to_string())
}

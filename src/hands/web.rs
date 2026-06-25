//! web is the "Hand" that turns the whole internet into a swappable Library. It queries a
//! self-hosted SearXNG metasearch instance over plain HTTP (so the Metis binary stays TLS-free —
//! SearXNG does the HTTPS to the upstream engines) and returns ranked results.
//!
//! Crucially, these results are NOT trusted raw: they become *evidence* that flows through the same
//! `ground -> generate -> verify -> cite -> abstain` discipline as local knowledge. The web is just
//! a Library too big to store, fetched on demand and verified like everything else. Sovereign (you
//! run the SearXNG), no API key, no vendor lock-in.

use serde::Deserialize;
use std::time::Duration;

/// One web result: a title, a snippet, and the source URL we cite.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct WebResult {
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub url: String,
    #[serde(default)]
    pub content: String,
}

/// search queries a SearXNG instance's JSON API and returns up to `n` results. `base` is the
/// SearXNG base URL (e.g. http://127.0.0.1:8888), taken from METIS_SEARCH_URL by the caller.
pub fn search(base: &str, query: &str, n: usize) -> Result<Vec<WebResult>, String> {
    let base = base.trim_end_matches('/');
    let agent = ureq::AgentBuilder::new()
        .timeout(Duration::from_secs(20))
        .build();
    let resp = agent
        .get(&format!("{base}/search"))
        .query("q", query)
        .query("format", "json")
        .query("safesearch", "0")
        .call()
        .map_err(|e| e.to_string())?;
    let body: SearxResponse = resp.into_json().map_err(|e: std::io::Error| e.to_string())?;
    Ok(body.results.into_iter().take(n).collect())
}

#[derive(Deserialize, Default)]
struct SearxResponse {
    #[serde(default)]
    results: Vec<WebResult>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_searxng_json() {
        let j = r#"{"query":"x","results":[
            {"title":"A","url":"http://a","content":"alpha"},
            {"title":"B","url":"http://b"}
        ]}"#;
        let r: SearxResponse = serde_json::from_str(j).unwrap();
        assert_eq!(r.results.len(), 2);
        assert_eq!(r.results[0].title, "A");
        assert_eq!(r.results[0].content, "alpha");
        assert_eq!(r.results[1].content, ""); // missing snippet defaults to empty, no panic
    }
}

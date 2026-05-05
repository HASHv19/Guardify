import { useState, useEffect } from "react";
import { predictText, checkHealth } from "./api";

const EXAMPLES = [
  "Tumhara attitude bohot cheap hai stop acting like clown",
  "You are such a loser get lost",
  "Let's meet tomorrow at the cafe",
  "Bhai tu pagal hai kya, kya bakwas kar raha hai",
  "What a beautiful sunset today I love this view!",
  "Nobody even likes you why do you post this trash",
  "I am so excited for the new movie coming out next week",
  "Abe gadhe chup kar teri aukaat nahi hai bolne ki",
  "Can you please send me the notes for tomorrow's lecture?",
  "You guys are pathetic and completely useless",
  "Mera naya laptop kal deliver ho jayega bhai",
  "Shut up no one cares about your stupid opinion",
  "Happy birthday! Have a great year ahead",
  "Tum itne ghatiya kaise ho sakte ho yaar",
  "Just finished a great workout feeling very healthy today"
];

function buildHighlightedText(text, flaggedTokens) {
  if (!text) {
    return [];
  }

  const escapedTokens = flaggedTokens
    .filter(Boolean)
    .map((token) => token.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));

  if (!escapedTokens.length) {
    return [{ type: "text", value: text }];
  }

  const matcher = new RegExp(`(${escapedTokens.join("|")})`, "gi");
  return text.split(matcher).filter(Boolean).map((part) => ({
    type: flaggedTokens.some((token) => token.toLowerCase() === part.toLowerCase()) ? "flag" : "text",
    value: part
  }));
}

export default function App() {
  const [text, setText] = useState(EXAMPLES[0]);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState(() => {
    try {
      const stored = localStorage.getItem("guardify_history");
      return stored ? JSON.parse(stored) : [];
    } catch (e) {
      return [];
    }
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [apiStatus, setApiStatus] = useState("Checking...");

  useEffect(() => {
    localStorage.setItem("guardify_history", JSON.stringify(history));
  }, [history]);

  useEffect(() => {
    async function initHealth() {
      try {
        const data = await checkHealth();
        setApiStatus(data.status === "ok" ? "Online" : "Offline");
      } catch (err) {
        setApiStatus("Offline");
      }
    }
    initHealth();
  }, []);

  async function handleSubmit(event) {
    event.preventDefault();
    setResult(null);
    setLoading(true);
    setError("");

    try {
      const next = await predictText(text);
      setResult(next);
      setHistory((current) => [next, ...current].slice(0, 5));
    } catch (err) {
      setError(err.message || "Prediction failed.");
    } finally {
      setLoading(false);
    }
  }

  const activeResult = result
    ? result
    : {
        label: "Standby",
        confidence: 0,
        probabilities: { Bullying: 0, "Non-Bullying": 0 },
        flagged_tokens: [],
        normalized_text: "",
        model_version: "Awaiting analysis",
        sub_category: null
      };

  const highlighted = buildHighlightedText(text, activeResult.flagged_tokens);
  const confidencePercent = Math.round(activeResult.confidence * 100);
  const confidenceLabel = confidencePercent < 50 ? "Low" : confidencePercent <= 80 ? "Medium" : "High";
  const isBullying = activeResult.label === "Bullying";
  const verdictTitle =
    activeResult.label === "Standby"
      ? "Awaiting Analysis"
      : isBullying
        ? "Cyberbullying Detected"
        : "No Bullying Detected";
  const verdictCopy = isBullying
    ? "Immediate moderation recommended based on Guardify protocol."
    : activeResult.label === "Non-Bullying"
      ? "No direct bullying pattern detected in the current sample."
      : "Submit text to generate a live Guardify analysis.";

  return (
    <>
      <div className="ambient ambient-top" />
      <div className="ambient ambient-bottom" />

      <header className="topbar">
        <div className="brand">
          <div className="brand-shield">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="brand-icon">
              <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
              <circle cx="12" cy="11" r="3" fill="var(--primary)" fillOpacity="0.2"/>
              <path d="M12 8v6" stroke="var(--primary)"/>
            </svg>
          </div>
          <div className="brand-copy">
            <span className="brand-name">Guardify</span>
            <span className="brand-tag">Moderation Dashboard</span>
          </div>
        </div>
        <div className="api-status">
          <span className={`status-dot ${apiStatus.replace("...", "").toLowerCase()}`}></span>
          <span className="status-text">API {apiStatus}</span>
        </div>
      </header>

      <main className="dashboard-shell">
        <section className="section-intro">
          <span className="section-kicker">Live Detection</span>
          <h1>Analyze Social Content</h1>
          <p className="section-copy">
            Submit English or Hinglish text and inspect the real Guardify model output, including verdict,
            confidence, normalized text, and flagged tokens.
          </p>
        </section>

        <section className="input-panel">
          <form className="composer-shell" onSubmit={handleSubmit}>
            <div className="composer-highlight" />
            <textarea
              id="text-input"
              aria-label="Text to analyze"
              rows="6"
              value={text}
              onChange={(event) => setText(event.target.value)}
              placeholder="Paste a tweet, comment, or social media post in English or Hinglish..."
            />
            <div className="composer-actions">
              <div className="example-pills">
                <button 
                  type="button" 
                  className="example-pill" 
                  onClick={() => {
                    let nextSample;
                    do {
                      nextSample = EXAMPLES[Math.floor(Math.random() * EXAMPLES.length)];
                    } while (nextSample === text);
                    setText(nextSample);
                    setResult(null);
                  }}
                >
                  🎲 Load Random Sample
                </button>
              </div>
              {text && (
                <button type="button" className="clear-button" onClick={() => { setText(""); setResult(null); }}>
                  Clear
                </button>
              )}
            </div>
            <div className="composer-footer">
              <span className="char-counter">{text.length} chars</span>
              <button className="analyze-button" type="submit" disabled={loading || !text.trim() || apiStatus === "Offline"}>
                {loading ? "Analyzing..." : "Analyze Content"}
              </button>
            </div>
            {error ? <p className="error-text">{error}</p> : null}
          </form>
        </section>

        {loading ? (
          <section className="analysis-panel">
            <div className="progress-loader">
              <div className="progress-bar-track">
                <div className="progress-bar-fill"></div>
              </div>
              <div className="progress-steps">
                <span className="progress-step active">⚙ Preprocessing text</span>
                <span className="progress-step">🔍 Extracting features</span>
                <span className="progress-step">🧠 Running SVM model</span>
                <span className="progress-step">✦ Generating verdict</span>
              </div>
              <p className="progress-label">Guardify Neural Analysis in Progress...</p>
            </div>
          </section>
        ) : (
          <>
            {(result) && (
              <>
            <section className="results-grid" aria-live="polite">
              <article className={`metric-card verdict-card ${isBullying ? "bullying" : activeResult.label === "Non-Bullying" ? "safe" : ""}`}>
                <div className="metric-head">
                  <span>Detection Verdict</span>
                  <span className="metric-icon">{isBullying ? "!" : activeResult.label === "Non-Bullying" ? "✓" : "~"}</span>
                </div>
                <div className="metric-body">
                  <h2 className="verdict-headline">
                    {verdictTitle}
                    {isBullying && activeResult.sub_category && (
                      <span className="sub-category-badge">{activeResult.sub_category}</span>
                    )}
                  </h2>
                  <p>{verdictCopy}</p>
                </div>
              </article>

              <article className="metric-card confidence-card">
                <div className="metric-head">
                  <span>Model Confidence</span>
                  <span className="metric-icon">◌</span>
                </div>
                <div className="confidence-body">
                  <div className={`confidence-badge ${confidenceLabel.toLowerCase()}`}>
                    <strong>{confidenceLabel} Confidence</strong>
                    <span>{confidencePercent}%</span>
                  </div>
                  <div className="confidence-copy">
                    <p>Model output confidence</p>
                    <span>{activeResult.model_version}</span>
                  </div>
                </div>
              </article>
            </section>

            <section className="analysis-panel">
              <div className="analysis-header">
                <span>Detailed Analysis</span>
              </div>

              <div className="analysis-content compact">
                <div className="analysis-block">
                  <h3>Original Text With Flagged Tokens</h3>
                  <div className="annotated-copy">
                    {highlighted.length ? (
                      highlighted.map((part, index) =>
                        part.type === "flag" ? (
                          <mark key={`${part.value}-${index}`}>{part.value}</mark>
                        ) : (
                          <span key={`${part.value}-${index}`}>{part.value}</span>
                        )
                      )
                    ) : (
                      <span>Run analysis to highlight abusive language found by Guardify.</span>
                    )}
                  </div>
                </div>

                <div className="detail-grid">
                  <div className="analysis-block">
                    <h3>Flagged Tokens</h3>
                    <div className="token-list">
                      {activeResult.flagged_tokens.length ? (
                        activeResult.flagged_tokens.map((token) => (
                          <span className="flag-chip" key={token}>
                            {token}
                          </span>
                        ))
                      ) : (
                        <p className="empty-copy">No explicit flagged tokens detected.</p>
                      )}
                    </div>
                  </div>

                  <div className="analysis-block">
                    <h3>Probabilities</h3>
                    <div className="stat-card-list">
                      <div className="stat-card">
                        <span>Bullying</span>
                        <strong>{(activeResult.probabilities.Bullying * 100).toFixed(1)}%</strong>
                      </div>
                      <div className="stat-card">
                        <span>Non-Bullying</span>
                        <strong>{(activeResult.probabilities["Non-Bullying"] * 100).toFixed(1)}%</strong>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="analysis-block">
                  <h3>Normalized Text</h3>
                  <div className="normalized-panel">
                    {activeResult.normalized_text || "Normalized text will appear here after analysis."}
                  </div>
                </div>
              </div>
            </section>
              </>
            )}
          </>
        )}

        <section className="history-panel">
          <div className="history-heading">
            <h2>Recent Analyses</h2>
            <p>Client-side session history for the latest Guardify scans.</p>
          </div>
          <div className="history-stream">
            {history.length ? (
              history.map((entry, index) => (
                <article className="history-tile" key={`${entry.normalized_text}-${index}`}>
                  <div className={`history-verdict ${entry.label === "Bullying" ? "bullying" : "safe"}`}>
                    {entry.label}
                  </div>
                  <p>{entry.normalized_text}</p>
                </article>
              ))
            ) : (
              <p className="empty-copy">No predictions yet.</p>
            )}
          </div>
        </section>
      </main>
    </>
  );
}

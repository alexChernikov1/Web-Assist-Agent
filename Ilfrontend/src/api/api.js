
// frontend/api.js
// ---------------------------------------------------------------------------
// Adds a quick “Verifying part/model …” message BEFORE the round‑trip to
// the backend so the UI never feels frozen.  Pass a callback that appends
// chat messages to your UI (e.g. addAssistantMessage) as the 2nd argument.
// ---------------------------------------------------------------------------

/**
 * Simple heuristics:
 *   • PartSelect parts    → “PS” followed by ≥5 digits  (e.g. PS11752778)
 *   • Model numbers       → any 6‑plus‑char alnum token with ≥2 letters +
 *                            ≥2 digits (e.g. WDT780SAEM1, WRX735SDHZ03)
 */
// api.js  (replace the previous version)
const PART_RE  = /\bPS\d{5,}\b/i;
const MODEL_RE = /\b[A-Z]{2,}[A-Z0-9\-]{2,}\d{2,}[A-Z0-9\-]*\b/;

/**
 * @param {string}   userQuery
 * @param {Function} pushProgress  – called once to show “Verifying …”
 * @returns {Promise<{role: string, content: string}>}
 */
export const getAIMessage = async (userQuery, pushProgress) => {
  // fire the progress note exactly once
  let progressShown = false;
  if (PART_RE.test(userQuery) || MODEL_RE.test(userQuery)) {
    pushProgress({
      role: "assistant",
      content: "Verifying part/model number — this might take a second…",
    });
    progressShown = true;
  }

  try {
    const response = await fetch("http://127.0.0.1:8000/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: userQuery }),
    });

    if (!response.ok) throw new Error("Backend error");

    const data = await response.json();

    // backend may echo the same “Verifying …” preface – remove it
    const cleaned = data.reply.replace(
      /^Verifying[^\n]*\n+/i,   // first “Verifying …” line & following blank line(s)
      ""
    );

    return { role: "assistant", content: cleaned, progressShown };
  } catch (err) {
    console.error("Backend error:", err);
    return { role: "assistant", content: "Error connecting to backend." };
  }
};




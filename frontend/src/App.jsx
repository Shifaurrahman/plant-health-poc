import { useState } from "react";

const API = "http://127.0.0.1:8000";
const card = { border: "1px solid #ddd", borderRadius: 8, padding: 12, marginBottom: 8 };

function Result({ data }) {
  // Only image matches are shown as reference photos
  const photos = data.retrieved_evidence.filter((e) => e.type === "image");

  return (
    <div style={{ display: "grid", gap: 20 }}>
      <section>
        <h2>Coverage: {data.coverage}</h2>
        <p>{data.next_step}</p>
      </section>

      <section>
        <h3>Candidate causes</h3>
        {data.candidate_causes.length === 0 && <p>No approved match found.</p>}
        {data.candidate_causes.map((c) => (
          <div key={c.name} style={card}>
            <strong>{c.name}</strong>
            <ul>{c.support.map((s) => <li key={s}>{s}</li>)}</ul>
            <em>{c.against_or_missing.join(" ")}</em>
          </div>
        ))}
      </section>

      {photos.length > 0 && (
        <section>
          <h3>Matched reference photos</h3>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(160px, 1fr))", gap: 12 }}>
            {photos.map((p) => (
              <figure key={p.id} style={{ margin: 0 }}>
                <img src={`${API}/images/${p.id}`} alt={p.title} style={{ width: "100%", borderRadius: 6 }} />
                <figcaption style={{ fontSize: 12 }}>
                  {p.title} · similarity {p.similarity.toFixed(2)}
                </figcaption>
              </figure>
            ))}
          </div>
        </section>
      )}

      {data.questions.length > 0 && (
        <section>
          <h3>Please add</h3>
          <ul>{data.questions.map((q) => <li key={q}>{q}</li>)}</ul>
        </section>
      )}

      <section>
        <h3>Limitations</h3>
        <ul>{data.limitations.map((l) => <li key={l}>{l}</li>)}</ul>
      </section>
    </div>
  );
}

export default function App() {
  const [caseId, setCaseId] = useState(null);
  const [text, setText] = useState("");
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // Create a case on first use and reuse it afterwards
  async function getCaseId() {
    if (caseId) return caseId;
    const res = await fetch(`${API}/cases`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ crop: "brinjal", region: "Sri Lanka", plant_part: "leaf" }),
    });
    if (!res.ok) throw new Error(`Case create failed (${res.status})`);
    const data = await res.json();
    setCaseId(data.case_id);
    return data.case_id;
  }

  async function handleSubmit(e) {
    e.preventDefault();
    setError("");
    setResult(null);
    setLoading(true);
    try {
      const id = await getCaseId();
      const form = new FormData();
      form.append("text", text.trim());
      if (file) form.append("files", file);

      const res = await fetch(`${API}/cases/${id}/messages`, { method: "POST", body: form });

      // 404 means the server restarted and the case is gone.
      // Clear the stored case ID so the next submit creates a new case.
      if (res.status === 404) {
        setCaseId(null);
        throw new Error("Case expired (the server may have restarted). Please submit again.");
      }
      if (!res.ok) throw new Error(`Request failed (${res.status}): ${await res.text()}`);

      setResult(await res.json());
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  // Enable the button only when there is text or a photo
  const canSubmit = !loading && (text.trim() || file);

  return (
    <div style={{ maxWidth: 900, margin: "0 auto", padding: 24, fontFamily: "system-ui, sans-serif" }}>
      <h1>Plant Health POC</h1>
      <p style={{ color: "#555" }}>Brinjal only. Decision support, not a diagnosis.</p>

      <form onSubmit={handleSubmit} style={{ display: "grid", gap: 12, marginBottom: 24 }}>
        <textarea
          rows={3}
          placeholder="e.g. brown spots on lower leaves"
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
        <input
          type="file"
          accept="image/jpeg,image/png,image/webp"
          onChange={(e) => setFile(e.target.files[0] ?? null)}
        />
        <button type="submit" disabled={!canSubmit}>
          {loading ? "Checking..." : "Ask"}
        </button>
      </form>

      {error && <div style={{ color: "#b00020", marginBottom: 16 }}>{error}</div>}
      {result && <Result data={result} />}
    </div>
  );
}
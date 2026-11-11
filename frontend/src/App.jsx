import { useEffect, useState } from "react";

const API = "http://127.0.0.1:8000";

const OUTCOME_TITLE = {
  match: "Visual match found",
  healthy_appearance: "Looks healthy compared with references",
  unknown_cause: "No single cause stands out",
  insufficient_image: "Photo not suitable for analysis",
  unsupported_crop: "Crop not supported",
  text_only_match: "Text match only",
  no_match: "No approved reference match",
  conflict: "Photo and reference records disagree",
};

function uploadUrl(path) {
  const parts = String(path).replace(/\\/g, "/").split("/");
  const caseId = parts.at(-2);
  const name = parts.at(-1);
  return `${API}/uploads/${caseId}/${name}`;
}

function OutcomeBanner({ outcome, route, headline }) {
  return (
    <section className="card banner">
      <strong>{headline || OUTCOME_TITLE[outcome] || outcome}</strong>
      {route && <div className="muted">Route: {route.replace("_", " ")}</div>}
    </section>
  );
}

function Signals({ signals }) {
  const sim = signals.retrieval_relevance.best_reference_similarity;
  return (
    <section className="card">
      <h3>How to read this result</h3>
      <ul>
        <li>
          <strong>Retrieval relevance:</strong>{" "}
          {sim == null ? "no photo match" : `closest reference photo scored ${sim.toFixed(2)} (visual only)`}.
          <div className="muted">{signals.retrieval_relevance.meaning}</div>
        </li>
        <li>
          <strong>Evidence coverage:</strong> {signals.evidence_coverage}
          <div className="muted">Whether approved records support this answer. Not a diagnosis score.</div>
        </li>
        <li>
          <strong>Diagnostic confidence:</strong> not assessed
          <div className="muted">{signals.diagnostic_confidence.meaning}</div>
        </li>
      </ul>
    </section>
  );
}

function EvidenceCard({ e }) {
  const isImage = e.type === "image";
  const sourceLine = [e.source_title ?? e.source_id, e.publisher, e.published_on].filter(Boolean).join(", ");
  const reviewLine = [e.review_status, e.label_basis, e.reviewed_by && `by ${e.reviewed_by}`].filter(Boolean).join(" · ");
  return (
    <figure style={{ margin: 0 }}>
      {isImage && (
        <img src={`${API}/images/${e.id}`} alt={e.title} />
      )}
      <figcaption style={{ fontSize: 12, marginTop: 4 }}>
        <strong>{e.title}</strong>
        <div>Source: {sourceLine || "not recorded"}</div>
        <div>Review: {reviewLine || "not recorded"}</div>
        {isImage && e.similarity != null && <div>Retrieval relevance: {e.similarity.toFixed(2)}</div>}
        {!isImage && e.matched_terms && <div>Matched words: {e.matched_terms.join(", ")}</div>}
        {e.channel && <div>Channel: {e.channel}</div>}
      </figcaption>
    </figure>
  );
}

function AssessmentBody({ data }) {
  const images = (data.retrieved_evidence || []).filter((e) => e.type === "image");
  const texts = (data.retrieved_evidence || []).filter((e) => e.type === "text");
  return (
    <div className="grid">
      <OutcomeBanner outcome={data.outcome} route={data.route} headline={data.headline} />
      {data.observations?.length > 0 && (
        <section className="card">
          <h3>What the photo shows</h3>
          <ul>{data.observations.map((o) => <li key={o}>{o}</li>)}</ul>
        </section>
      )}
      <section className="card">
        <h3>Coverage: {data.coverage}</h3>
        <p>{data.next_step}</p>
      </section>
      {data.signals && <Signals signals={data.signals} />}
      <section>
        <h3>Candidate causes</h3>
        {(data.candidate_causes || []).length === 0 && <p>No candidate cause from the approved references.</p>}
        {(data.candidate_causes || []).map((c) => (
          <div key={c.name} className="card">
            <strong>{c.name}</strong>
            {c.fit && <div className="muted">Fit with photo: {c.fit}</div>}
            <div className="muted">Supporting observations</div>
            <ul>{(c.support || []).map((s) => <li key={s}>{s}</li>)}</ul>
            {c.against_or_missing?.length > 0 && (
              <>
                <div className="muted">Against or missing</div>
                <ul>{c.against_or_missing.map((s) => <li key={s}>{s}</li>)}</ul>
              </>
            )}
            {c.general_knowledge?.length > 0 && (
              <>
                <div className="muted warn">Model general knowledge (not from reviewed records)</div>
                <ul className="warn">{c.general_knowledge.map((s) => <li key={s}>{s}</li>)}</ul>
              </>
            )}
          </div>
        ))}
      </section>
      {data.alternatives?.length > 0 && (
        <section className="card">
          <h3>Alternatives</h3>
          <ul>{data.alternatives.map((a) => <li key={a}>{a}</li>)}</ul>
        </section>
      )}
      {images.length > 0 && (
        <section>
          <h3>Reference comparison</h3>
          <div className="photos">{images.map((e) => <EvidenceCard key={e.id} e={e} />)}</div>
        </section>
      )}
      {texts.length > 0 && (
        <section>
          <h3>Cited records</h3>
          <div className="grid">{texts.map((e) => <div key={e.id} className="card"><EvidenceCard e={e} /></div>)}</div>
        </section>
      )}
      {data.questions?.length > 0 && (
        <section className="card">
          <h3>Please add</h3>
          <ul>{data.questions.map((q) => <li key={q}>{q}</li>)}</ul>
        </section>
      )}
      <section className="card">
        <h3>Limitations</h3>
        <ul>{(data.limitations || []).map((l) => <li key={l}>{l}</li>)}</ul>
      </section>
      <details className="card">
        <summary>Technical details (versions and timing)</summary>
        <pre style={{ fontSize: 12, overflowX: "auto" }}>
          {JSON.stringify({ versions: data.versions, latency_ms: data.latency_ms, policy_removals: data.policy_removals }, null, 2)}
        </pre>
      </details>
    </div>
  );
}

function FeedbackForm({ caseId, turnIndex }) {
  const [helpful, setHelpful] = useState(null);
  const [disputed, setDisputed] = useState(false);
  const [helped, setHelped] = useState(null);
  const [comment, setComment] = useState("");
  const [status, setStatus] = useState("");

  async function submit() {
    setStatus("Sending...");
    try {
      const res = await fetch(`${API}/cases/${caseId}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          turn_index: turnIndex,
          helpful,
          disputed,
          next_action_helped: helped,
          comment: comment || null,
        }),
      });
      if (!res.ok) throw new Error(`status ${res.status}`);
      setStatus("Thank you. Your feedback was saved.");
    } catch (err) {
      setStatus(`Could not save feedback (${err.message}).`);
    }
  }

  return (
    <section className="card">
      <h3>Was this answer useful?</h3>
      <div style={{ display: "flex", gap: 8, marginBottom: 8 }}>
        <button type="button" className="secondary" onClick={() => setHelpful(true)} disabled={helpful === true}>Helpful</button>
        <button type="button" className="secondary" onClick={() => setHelpful(false)} disabled={helpful === false}>Not helpful</button>
      </div>
      <label>
        <input type="checkbox" checked={disputed} onChange={(e) => setDisputed(e.target.checked)} />{" "}
        Something in this answer looks wrong
      </label>
      <label style={{ marginTop: 8 }}>
        Did the next step help?
        <select value={helped ?? ""} onChange={(e) => setHelped(e.target.value === "" ? null : e.target.value === "yes")}>
          <option value="">Not sure</option>
          <option value="yes">Yes</option>
          <option value="no">No</option>
        </select>
      </label>
      <textarea rows={2} placeholder="Optional comment" value={comment} onChange={(e) => setComment(e.target.value)} />
      <button type="button" onClick={submit} disabled={helpful === null} style={{ marginTop: 8 }}>Send feedback</button>
      {status && <div className="muted">{status}</div>}
    </section>
  );
}

function ReviewForm({ caseId, turnIndex, onSaved }) {
  const [reviewer, setReviewer] = useState("");
  const [decision, setDecision] = useState("uncertain");
  const [method, setMethod] = useState("none");
  const [publish, setPublish] = useState("exclude");
  const [corrected, setCorrected] = useState("");
  const [notes, setNotes] = useState("");
  const [status, setStatus] = useState("");

  async function submit() {
    setStatus("Sending...");
    try {
      const res = await fetch(`${API}/cases/${caseId}/reviews`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          turn_index: turnIndex,
          reviewer,
          decision,
          confirmation_method: method,
          publish_decision: publish,
          corrected_condition: corrected || null,
          notes: notes || null,
        }),
      });
      if (!res.ok) throw new Error(`status ${res.status}`);
      setStatus("Review saved. Unresolved cases stay out of trusted references.");
      onSaved?.();
    } catch (err) {
      setStatus(`Could not save review (${err.message}).`);
    }
  }

  return (
    <details className="card">
      <summary>Expert review</summary>
      <div className="grid" style={{ marginTop: 8 }}>
        <input placeholder="Reviewer name" value={reviewer} onChange={(e) => setReviewer(e.target.value)} />
        <select value={decision} onChange={(e) => setDecision(e.target.value)}>
          <option value="confirmed">Confirmed</option>
          <option value="corrected">Corrected</option>
          <option value="uncertain">Uncertain</option>
          <option value="needs_lab_test">Needs lab test</option>
        </select>
        <select value={method} onChange={(e) => setMethod(e.target.value)}>
          <option value="none">Confirmation method: none</option>
          <option value="expert_visual">Expert visual</option>
          <option value="field_inspection">Field inspection</option>
          <option value="lab">Lab</option>
        </select>
        <select value={publish} onChange={(e) => setPublish(e.target.value)}>
          <option value="exclude">Keep out of trusted index</option>
          <option value="candidate_reference">Candidate reference</option>
          <option value="correction">Correction to existing record</option>
        </select>
        <input placeholder="Corrected condition (if any)" value={corrected} onChange={(e) => setCorrected(e.target.value)} />
        <textarea rows={2} placeholder="Notes" value={notes} onChange={(e) => setNotes(e.target.value)} />
        <button type="button" onClick={submit} disabled={!reviewer.trim()}>Save review</button>
        {status && <div className="muted">{status}</div>}
      </div>
    </details>
  );
}

function NewAssessment({ onCreated }) {
  const [text, setText] = useState("");
  const [files, setFiles] = useState([]);
  const [crop, setCrop] = useState("brinjal");
  const [region, setRegion] = useState("Sri Lanka");
  const [duration, setDuration] = useState("");
  const [part, setPart] = useState("leaf");
  const [treatment, setTreatment] = useState("");
  const [consent, setConsent] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [guidance, setGuidance] = useState([]);

  useEffect(() => {
    fetch(`${API}/guidance/photos`)
      .then((r) => r.json())
      .then((d) => setGuidance(d.views || []))
      .catch(() => {});
  }, []);

  async function handleSubmit(e) {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      const created = await fetch(`${API}/cases`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          crop,
          region,
          symptom_duration: duration || null,
          plant_part: part || null,
          recent_watering_or_treatment: treatment || null,
          consent_reuse_photos: consent,
        }),
      });
      if (!created.ok) throw new Error(`Case create failed (${created.status})`);
      const data = await created.json();
      const form = new FormData();
      form.append("text", text.trim());
      [...files].slice(0, 4).forEach((f) => form.append("files", f));
      const res = await fetch(`${API}/cases/${data.case_id}/messages`, { method: "POST", body: form });
      if (!res.ok) {
        const body = await res.json().catch(() => null);
        throw new Error(body?.detail ?? `Request failed (${res.status})`);
      }
      await res.json();
      onCreated(data.case_id);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  const canSubmit = !loading && (text.trim() || files.length);
  const photosNeedConsent = files.length > 0 && !consent;

  return (
    <form onSubmit={handleSubmit} className="grid">
      <section className="card">
        <h2>New assessment</h2>
        <p className="muted">Upload 1–4 photos and a question. Crop and context are optional but improve routing.</p>
        <textarea rows={3} placeholder="e.g. What might be causing these brown patches? The lower leaves were affected first." value={text} onChange={(e) => setText(e.target.value)} />
        <label>
          Photos (JPEG, PNG or WebP, up to 8 MB each)
          <input type="file" accept="image/jpeg,image/png,image/webp" multiple onChange={(e) => setFiles([...e.target.files].slice(0, 4))} />
        </label>
        {files.length > 0 && <div className="muted">{files.length} file(s) selected</div>}
      </section>
      <section className="card">
        <h3>Optional context</h3>
        <div className="grid two">
          <label>Crop<select value={crop} onChange={(e) => setCrop(e.target.value)}>
            <option value="brinjal">Brinjal</option>
            <option value="tomato">Tomato</option>
            <option value="other">Other (unsupported in this pilot)</option>
          </select></label>
          <label>Region<input type="text" value={region} onChange={(e) => setRegion(e.target.value)} /></label>
          <label>Symptom duration<input type="text" value={duration} onChange={(e) => setDuration(e.target.value)} placeholder="e.g. about 2 weeks" /></label>
          <label>Affected plant part<input type="text" value={part} onChange={(e) => setPart(e.target.value)} /></label>
        </div>
        <label style={{ marginTop: 8 }}>Recent watering or treatment
          <input type="text" value={treatment} onChange={(e) => setTreatment(e.target.value)} />
        </label>
        <label style={{ marginTop: 8 }}>
          <input type="checkbox" checked={consent} onChange={(e) => setConsent(e.target.checked)} />{" "}
          I agree that these photos may be stored for this case. Reuse as reference material needs a later expert review.
        </label>
      </section>
      <section className="card">
        <h3>Photo guidance</h3>
        <ul>{guidance.map((g) => <li key={g.question}><strong>{g.question}</strong> {g.why}</li>)}</ul>
      </section>
      {error && <div className="error">{error}</div>}
      <button type="submit" disabled={!canSubmit || photosNeedConsent}>
        {loading ? "Checking..." : "Ask"}
      </button>
      {photosNeedConsent && <div className="muted">Tick the consent box to upload photos.</div>}
    </form>
  );
}

function CaseScreen({ caseId, onMissing }) {
  const [caseData, setCaseData] = useState(null);
  const [text, setText] = useState("");
  const [files, setFiles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [reviewMsg, setReviewMsg] = useState("");

  async function load() {
    const res = await fetch(`${API}/cases/${caseId}`);
    if (res.status === 404) {
      onMissing?.();
      return;
    }
    if (!res.ok) throw new Error(`Could not load case (${res.status})`);
    setCaseData(await res.json());
  }

  useEffect(() => {
    load().catch((e) => setError(e.message));
  }, [caseId]);

  async function sendFollowUp(e) {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      const form = new FormData();
      form.append("text", text.trim());
      [...files].slice(0, 4).forEach((f) => form.append("files", f));
      const res = await fetch(`${API}/cases/${caseId}/messages`, { method: "POST", body: form });
      if (res.status === 404) {
        onMissing?.();
        throw new Error("Case expired. Start a new assessment.");
      }
      if (!res.ok) {
        const body = await res.json().catch(() => null);
        throw new Error(body?.detail ?? `Request failed (${res.status})`);
      }
      setText("");
      setFiles([]);
      await load();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function requestReview() {
    const res = await fetch(`${API}/cases/${caseId}/request-review`, { method: "POST" });
    if (res.ok) {
      setReviewMsg("Expert review requested.");
      load();
    } else {
      setReviewMsg("Could not request review.");
    }
  }

  if (!caseData) return <p className="muted">{error || "Loading case..."}</p>;
  const turns = caseData.turns || [];
  const latest = turns.at(-1);

  return (
    <div className="grid">
      <section className="card">
        <h2>Case {caseId}</h2>
        <p className="muted">
          Crop: {caseData.meta?.crop || "unspecified"} · Region: {caseData.meta?.region || "unspecified"} ·
          Part: {caseData.meta?.plant_part || "unspecified"}
        </p>
        <button type="button" className="secondary" onClick={requestReview}>Request expert review</button>
        {reviewMsg && <div className="muted">{reviewMsg}</div>}
      </section>

      {turns.map((t, i) => (
        <section key={t.at} className="history">
          <div className="muted">{t.at} · “{t.text || "(photos only)"}”</div>
          {t.images?.length > 0 && (
            <div className="thumbs">
              {t.images.map((p) => <img key={p} src={uploadUrl(p)} alt="case upload" />)}
            </div>
          )}
          {i === turns.length - 1 ? (
            <>
              <AssessmentBody data={t.assessment} />
              <FeedbackForm caseId={caseId} turnIndex={t.assessment.turn_index ?? i} />
              <ReviewForm caseId={caseId} turnIndex={t.assessment.turn_index ?? i} onSaved={load} />
            </>
          ) : (
            <details className="card">
              <summary>Previous assessment: {t.assessment.headline || OUTCOME_TITLE[t.assessment.outcome] || t.assessment.outcome}</summary>
              <AssessmentBody data={t.assessment} />
            </details>
          )}
        </section>
      ))}

      <form onSubmit={sendFollowUp} className="card grid">
        <h3>Continue this case</h3>
        <textarea rows={2} placeholder="Answer a follow-up, or describe what changed" value={text} onChange={(e) => setText(e.target.value)} />
        <input type="file" accept="image/jpeg,image/png,image/webp" multiple onChange={(e) => setFiles([...e.target.files].slice(0, 4))} />
        {error && <div className="error">{error}</div>}
        <button type="submit" disabled={loading || (!text.trim() && files.length === 0)}>
          {loading ? "Updating..." : "Send follow-up"}
        </button>
      </form>
      {latest && <p className="muted">Previous results stay available with timestamps.</p>}
    </div>
  );
}

function QueueScreen({ onOpen }) {
  const [items, setItems] = useState([]);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch(`${API}/reviews/queue`)
      .then((r) => r.json())
      .then((d) => setItems(d.items || []))
      .catch((e) => setError(e.message));
  }, []);

  return (
    <section className="card">
      <h2>Expert queue</h2>
      <p className="muted">Prioritised by dispute, requested review, missing knowledge and unreviewed cases.</p>
      {error && <div className="error">{error}</div>}
      <table>
        <thead>
          <tr><th>Priority</th><th>Case</th><th>Outcome</th><th>Coverage</th><th>Reasons</th></tr>
        </thead>
        <tbody>
          {items.map((it) => (
            <tr key={it.case_id}>
              <td className="priority">{it.priority}</td>
              <td><button type="button" className="secondary" onClick={() => onOpen(it.case_id)}>{it.case_id}</button></td>
              <td>{it.last_outcome || "—"}</td>
              <td>{it.coverage || "—"}</td>
              <td>{(it.reasons || []).join(", ")}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {items.length === 0 && <p className="muted">No cases yet.</p>}
    </section>
  );
}

function KnowledgeScreen() {
  const [records, setRecords] = useState([]);
  const [register, setRegister] = useState([]);
  const [status, setStatus] = useState("");

  async function load() {
    const [kr, cov] = await Promise.all([
      fetch(`${API}/knowledge/records`).then((r) => r.json()),
      fetch(`${API}/knowledge/coverage`).then((r) => r.json()),
    ]);
    setRecords(kr.records || []);
    setRegister(cov.register || []);
  }

  useEffect(() => { load().catch((e) => setStatus(e.message)); }, []);

  async function setReview(recordId, review_status) {
    const reviewed_by = review_status === "approved" ? "in-app-reviewer" : undefined;
    const res = await fetch(`${API}/knowledge/records/${recordId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ review_status, reviewed_by }),
    });
    if (!res.ok) {
      setStatus(`Could not update ${recordId}`);
      return;
    }
    setStatus(`Updated ${recordId} to ${review_status}. Restart retrieval uses this immediately for text; image index still needs approved classes.`);
    load();
  }

  return (
    <div className="grid">
      <section className="card">
        <h2>Knowledge manager</h2>
        <p className="muted">Only approved, non-retired records enter trusted retrieval. Missing fields stay empty; the model must not fill them in as facts.</p>
        {status && <div className="muted">{status}</div>}
        <table>
          <thead>
            <tr><th>Record</th><th>Condition</th><th>Status</th><th>Source</th><th>Actions</th></tr>
          </thead>
          <tbody>
            {records.map((r) => (
              <tr key={r.record_id}>
                <td>{r.record_id}</td>
                <td>{r.condition}</td>
                <td>{r.review_status} / {r.publication_status}</td>
                <td>{r.source_id}</td>
                <td>
                  <button type="button" className="secondary" onClick={() => setReview(r.record_id, "approved")}>Approve</button>{" "}
                  <button type="button" className="secondary" onClick={() => setReview(r.record_id, "retired")}>Retire</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
      <section className="card">
        <h3>Coverage register</h3>
        <table>
          <thead>
            <tr><th>Crop</th><th>Condition</th><th>Text</th><th>Images</th><th>Management</th><th>Weak eval</th></tr>
          </thead>
          <tbody>
            {register.map((row) => (
              <tr key={`${row.crop}-${row.condition}`}>
                <td>{row.crop}</td>
                <td>{row.condition}</td>
                <td>{row.has_symptoms ? "yes" : "gap"}</td>
                <td>{row.image_count}</td>
                <td>{row.has_management ? "yes" : "gap"}</td>
                <td>{row.weak_in_evaluation ? "yes" : "no"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  );
}

function DashboardScreen() {
  const [data, setData] = useState(null);
  const [gaps, setGaps] = useState([]);
  const [evalResult, setEvalResult] = useState(null);

  useEffect(() => {
    fetch(`${API}/dashboard`).then((r) => r.json()).then(setData).catch(() => {});
    fetch(`${API}/knowledge/gaps`).then((r) => r.json()).then((d) => setGaps(d.items || [])).catch(() => {});
  }, []);

  async function runEval() {
    const res = await fetch(`${API}/eval/run`, { method: "POST" });
    setEvalResult(await res.json());
  }

  if (!data) return <p className="muted">Loading dashboard...</p>;
  return (
    <div className="grid">
      <section className="card">
        <h2>Quality dashboard</h2>
        <ul>
          <li>Cases: {data.cases} · Turns: {data.turns}</li>
          <li>Reviews: {data.reviews} · Review requested: {data.review_requested} · Disputed: {data.disputed}</li>
          <li>Approved records: {data.approved_records} · Pending: {data.pending_records} · Gaps logged: {data.knowledge_gaps}</li>
          <li>Latency median / p95 (ms): {data.latency_ms?.median ?? "—"} / {data.latency_ms?.p95 ?? "—"}</li>
          <li>Policy removals: {data.policy_removals}</li>
        </ul>
        <p className="muted">Outcomes: {JSON.stringify(data.outcomes)}</p>
        <p className="muted">Coverage: {JSON.stringify(data.coverage)}</p>
        <button type="button" className="secondary" onClick={runEval}>Run evaluation gates</button>
        {evalResult && <pre style={{ fontSize: 12, overflowX: "auto" }}>{JSON.stringify(evalResult, null, 2)}</pre>}
      </section>
      <section className="card">
        <h3>Knowledge gaps</h3>
        {gaps.length === 0 && <p className="muted">None logged yet.</p>}
        <ul>
          {gaps.map((g, i) => (
            <li key={`${g.at}-${i}`}>{g.at}: {g.crop} · {g.outcome} · {g.question || "(no text)"}</li>
          ))}
        </ul>
      </section>
    </div>
  );
}

export default function App() {
  const [screen, setScreen] = useState("assess");
  const [caseId, setCaseId] = useState(null);

  function openCase(id) {
    setCaseId(id);
    setScreen("case");
  }

  return (
    <div className="app">
      <header className="header">
        <h1>Plant Health Assistant</h1>
        <p className="lede">Decision support for plant health questions. Not a confirmed diagnosis. Brinjal pilot.</p>
      </header>
      <nav className="nav">
        {[
          ["assess", "New assessment"],
          ["case", "Case"],
          ["queue", "Expert queue"],
          ["knowledge", "Knowledge"],
          ["dashboard", "Dashboard"],
        ].map(([id, label]) => (
          <button key={id} type="button" className={screen === id ? "active" : ""} onClick={() => setScreen(id)}>
            {label}
          </button>
        ))}
      </nav>
      {screen === "assess" && <NewAssessment onCreated={openCase} />}
      {screen === "case" && (caseId
        ? <CaseScreen caseId={caseId} onMissing={() => { setCaseId(null); setScreen("assess"); }} />
        : <p className="muted">Create an assessment, or open a case from the expert queue.</p>)}
      {screen === "queue" && <QueueScreen onOpen={openCase} />}
      {screen === "knowledge" && <KnowledgeScreen />}
      {screen === "dashboard" && <DashboardScreen />}
    </div>
  );
}
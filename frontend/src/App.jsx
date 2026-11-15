import { useCallback, useEffect, useMemo, useState } from "react";
import axios from "axios";

const api = axios.create({ baseURL: "/api", timeout: 30000 });
api.interceptors.request.use((config) => {
  const token = sessionStorage.getItem("admin_token");
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

const navItems = [
  ["dashboard", "Overview", "▦"],
  ["students", "Students", "♙"],
  ["upload", "Upload document", "↑"],
  ["documents", "Documents", "▤"],
  ["results", "Verification results", "◉"],
  ["model-lab", "ML Performance", "⌁"],
  ["xai", "Explainable AI (SHAP)", "⚲"],
  ["admin-users", "Admin Management", "⚙"],
  ["reviews", "Admin review", "✓"],
  ["history", "History", "◷"],
  ["audit", "Audit log", "≡"],
  ["settings", "Settings", "🔧"],
  ["profile", "Admin profile", "◎"],
];

const documentTypes = [
  ["marks_card", "Academic marks card / Statement of grades"],
  ["transcript", "Official academic transcript"],
  ["degree", "Degree / Diploma certificate"],
  ["sslc", "Secondary school certificate (SSLC/10th)"],
  ["puc", "Higher secondary certificate (PUC/12th)"],
  ["transfer", "Transfer / Migration certificate"],
  ["other", "Other academic record"],
];

function messageFrom(error) {
  return error?.response?.data?.detail || error?.message || "Something went wrong. Please try again.";
}

function dateLabel(value) {
  if (!value) return "—";
  return new Date(value).toLocaleString();
}

function StatusBadge({ value }) {
  const text = (value || "pending").replaceAll("_", " ");
  return <span className={`badge badge-${value || "pending"}`}>{text}</span>;
}

function Risk({ value, score }) {
  return (
    <span className={`risk risk-${value || "unknown"}`}>
      <i />
      {value ? `${value} risk${score !== undefined ? ` · ${score}/100` : ""}` : "Not assessed"}
    </span>
  );
}

function Search({ value, onChange }) {
  return (
    <label className="search-box">
      <span>⌕</span>
      <input placeholder="Search records…" value={value} onChange={(e) => onChange(e.target.value)} />
    </label>
  );
}

function StatusFilter({ value, onChange }) {
  return (
    <select className="status-filter" aria-label="Filter by review status" value={value} onChange={(e) => onChange(e.target.value)}>
      <option value="all">All statuses</option>
      <option value="pending">Pending</option>
      <option value="verified">Reviewed</option>
      <option value="suspicious">Flagged</option>
    </select>
  );
}

function EmptyRow({ columns, text }) {
  return (
    <tr>
      <td colSpan={columns} className="empty-cell">{text}</td>
    </tr>
  );
}

function formatMetric(value) {
  return Number.isFinite(value) ? value.toFixed(3) : "—";
}

function Login({ onLogin }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(event) {
    event.preventDefault();
    setError("");
    setBusy(true);
    try {
      const { data } = await api.post("/auth/login", { username, password });
      sessionStorage.setItem("admin_token", data.access_token);
      onLogin();
    } catch (requestError) {
      setError(messageFrom(requestError));
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="login-shell">
      <section className="login-card">
        <div className="brand-mark">CV</div>
        <p className="eyebrow">COLLEGE ADMINISTRATION</p>
        <h1>Welcome back</h1>
        <p className="muted">Sign in to review and verify academic documents.</p>
        <form className="stack-form login-form" onSubmit={submit}>
          <label>
            Administrator username
            <input autoComplete="username" required value={username} onChange={(e) => setUsername(e.target.value)} />
          </label>
          <label>
            Password
            <input type="password" autoComplete="current-password" required value={password} onChange={(e) => setPassword(e.target.value)} />
          </label>
          {error && <div className="alert error">{error}</div>}
          <button className="button primary full" disabled={busy}>{busy ? "Signing in…" : "Sign in"}</button>
        </form>
        <p className="login-foot">Default demo account: <code>demo_admin</code> / <code>AdminPassword@2026</code></p>
      </section>
    </main>
  );
}

function App() {
  const [authenticated, setAuthenticated] = useState(Boolean(sessionStorage.getItem("admin_token")));
  const [active, setActive] = useState("dashboard");
  const [stats, setStats] = useState(null);
  const [students, setStudents] = useState([]);
  const [documents, setDocuments] = useState([]);
  const [admin, setAdmin] = useState(null);
  const [auditLogs, setAuditLogs] = useState([]);
  const [mlSummary, setMlSummary] = useState(null);
  const [health, setHealth] = useState(null);
  const [adminUsers, setAdminUsers] = useState([]);
  const [query, setQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [selected, setSelected] = useState(null);
  const [history, setHistory] = useState([]);
  const [studentForm, setStudentForm] = useState({ full_name: "", email: "", registration_number: "" });
  const [editingStudentId, setEditingStudentId] = useState(null);
  const [uploadForm, setUploadForm] = useState({ student_id: "", document_type: "marks_card", file: null });

  const logout = useCallback(() => {
    sessionStorage.removeItem("admin_token");
    setAuthenticated(false);
    setAdmin(null);
  }, []);

  const refresh = useCallback(async () => {
    setError("");
    try {
      const [statsResponse, studentsResponse, documentsResponse, adminResponse] = await Promise.all([
        api.get("/dashboard/stats"),
        api.get("/students"),
        api.get("/documents"),
        api.get("/auth/me"),
      ]);
      setStats(statsResponse.data);
      setStudents(studentsResponse.data);
      setDocuments(documentsResponse.data);
      setAdmin(adminResponse.data);
    } catch (requestError) {
      if (requestError?.response?.status === 401) logout();
      else setError(messageFrom(requestError));
    }
  }, [logout]);

  useEffect(() => {
    if (authenticated) refresh();
  }, [authenticated, refresh]);

  useEffect(() => {
    if (authenticated && active === "history" && documents.length && !selected) {
      setSelected(documents[0]);
    }
  }, [authenticated, active, documents, selected]);

  useEffect(() => {
    if (!selected || active !== "history") return;
    api.get(`/documents/${selected.id}/history`)
      .then(({ data }) => setHistory(data))
      .catch((requestError) => setError(messageFrom(requestError)));
  }, [active, selected]);

  useEffect(() => {
    if (authenticated && active === "audit" && admin?.role === "admin") {
      api.get("/admin/audit-logs")
        .then(({ data }) => setAuditLogs(data))
        .catch((requestError) => setError(messageFrom(requestError)));
    }
  }, [active, authenticated, admin]);

  useEffect(() => {
    if (authenticated && (active === "model-lab" || active === "xai")) {
      api.get("/ml/summary")
        .then(({ data }) => setMlSummary(data))
        .catch((requestError) => setError(messageFrom(requestError)));
    }
  }, [active, authenticated]);

  useEffect(() => {
    if (authenticated && active === "admin-users" && admin?.role === "admin") {
      api.get("/admin/users")
        .then(({ data }) => setAdminUsers(data))
        .catch((requestError) => setError(messageFrom(requestError)));
    }
  }, [active, authenticated, admin]);

  useEffect(() => {
    if (authenticated && active === "settings") {
      api.get("/health")
        .then(({ data }) => setHealth(data))
        .catch((requestError) => setError(messageFrom(requestError)));
    }
  }, [active, authenticated]);

  const filteredDocuments = useMemo(() => {
    let rows = documents;
    if (active === "reviews") rows = rows.filter((item) => item.status === "pending" || item.status === "suspicious");
    if (statusFilter !== "all") rows = rows.filter((item) => item.status === statusFilter);
    if (query.trim()) {
      const search = query.trim().toLowerCase();
      rows = rows.filter((item) =>
        [item.original_filename, item.student_name, item.student_registration_number, item.document_type]
          .some((value) => String(value || "").toLowerCase().includes(search)),
      );
    }
    return rows;
  }, [active, documents, query, statusFilter]);

  function feedback(success, text) {
    setError("");
    setNotice(success ? text : "");
    if (!success) setError(text);
    window.setTimeout(() => setNotice(""), 5000);
  }

  async function addStudent(event) {
    event.preventDefault();
    setBusy(true);
    try {
      const payload = {
        ...studentForm,
        email: studentForm.email.trim() || null,
        registration_number: studentForm.registration_number.trim() || null,
      };
      if (editingStudentId) await api.put(`/students/${editingStudentId}`, payload);
      else await api.post("/students", payload);
      setStudentForm({ full_name: "", email: "", registration_number: "" });
      setEditingStudentId(null);
      feedback(true, editingStudentId ? "Student record updated." : "Student record added.");
      await refresh();
    } catch (requestError) {
      feedback(false, messageFrom(requestError));
    } finally {
      setBusy(false);
    }
  }

  function editStudent(student) {
    setEditingStudentId(student.id);
    setStudentForm({
      full_name: student.full_name,
      email: student.email || "",
      registration_number: student.registration_number || "",
    });
  }

  async function deleteStudent(student) {
    if (!window.confirm(`Delete ${student.full_name}? Students with uploaded documents cannot be deleted.`)) return;
    try {
      await api.delete(`/students/${student.id}`);
      feedback(true, "Student record deleted.");
      await refresh();
    } catch (requestError) {
      feedback(false, messageFrom(requestError));
    }
  }

  async function uploadDocument(event) {
    event.preventDefault();
    if (!uploadForm.file || !uploadForm.student_id) {
      feedback(false, "Choose a student and a document file.");
      return;
    }
    const body = new FormData();
    body.append("student_id", uploadForm.student_id);
    body.append("document_type", uploadForm.document_type);
    body.append("file", uploadForm.file);
    setBusy(true);
    try {
      const { data } = await api.post("/documents/upload", body, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setUploadForm((current) => ({ ...current, file: null }));
      event.target.reset();
      feedback(true, `Document uploaded. Risk assessment: ${data.latest_result?.risk_level || "pending"} (${data.latest_result?.risk_score}/100).`);
      setSelected(data);
      await refresh();
      setActive("results");
    } catch (requestError) {
      feedback(false, messageFrom(requestError));
    } finally {
      setBusy(false);
    }
  }

  async function runVerification(item) {
    setBusy(true);
    try {
      const { data } = await api.post(`/documents/${item.id}/verify`);
      setSelected(data);
      feedback(true, "Verification indicators refreshed. Human review is still required.");
      await refresh();
    } catch (requestError) {
      feedback(false, messageFrom(requestError));
    } finally {
      setBusy(false);
    }
  }

  async function downloadDocument(item) {
    try {
      const { data } = await api.get(`/documents/${item.id}/file`, { responseType: "blob" });
      const url = URL.createObjectURL(data);
      const link = document.createElement("a");
      link.href = url;
      link.download = item.original_filename;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (requestError) {
      feedback(false, messageFrom(requestError));
    }
  }

  async function review(item, decision) {
    const note = window.prompt(`Optional note for this ${decision} decision:`, "");
    if (note === null) return;
    setBusy(true);
    try {
      await api.post(`/documents/${item.id}/review`, { decision, note });
      feedback(true, `Document marked ${decision}.`);
      await refresh();
      if (selected?.id === item.id) setSelected({ ...item, status: decision });
    } catch (requestError) {
      feedback(false, messageFrom(requestError));
    } finally {
      setBusy(false);
    }
  }

  async function openDocument(item) {
    setSelected(item);
    try {
      const { data } = await api.get(`/documents/${item.id}`);
      setSelected(data);
    } catch (requestError) {
      feedback(false, messageFrom(requestError));
    }
  }

  if (!authenticated) return <Login onLogin={() => setAuthenticated(true)} />;

  const visibleNavItems = navItems.filter(([key]) => admin?.role === "admin" || !["students", "upload", "audit", "admin-users"].includes(key));
  const title = navItems.find(([key]) => key === active)?.[1] || "Overview";

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark small">CV</div>
          <div><strong>Campus Verify</strong><span>ACADEMIC VERIFICATION</span></div>
        </div>
        <div className="nav-caption">WORKSPACE</div>
        <nav>
          {visibleNavItems.map(([key, label, icon]) => (
            <button
              key={key}
              className={`nav-link ${active === key ? "active" : ""}`}
              onClick={() => { setActive(key); setQuery(""); setStatusFilter("all"); setError(""); }}
            >
              <span className="nav-icon">{icon}</span>{label}
              {key === "reviews" && stats?.pending_documents > 0 && <span className="nav-count">{stats.pending_documents}</span>}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="secure-label"><span className="secure-dot" />{admin?.role === "reviewer" ? "Reviewer workspace" : "Administrator workspace"}</div>
          <button className="nav-link logout" onClick={logout}><span className="nav-icon">↪</span>Sign out</button>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div className="mobile-brand"><div className="brand-mark small">CV</div><strong>Campus Verify</strong></div>
          <div className="breadcrumb">Workspace <span>/</span> <strong>{title}</strong></div>
          <div className="admin-chip">
            <span className="avatar">{(admin?.username || "A").slice(0, 1).toUpperCase()}</span>
            <span>{admin?.username || "Staff"} · {admin?.role || "staff"}</span>
          </div>
        </header>

        <div className="page-content">
          <div className="page-heading">
            <div>
              <p className="eyebrow">INTELLIGENT DOCUMENT VERIFICATION & FRAUD DETECTION</p>
              <h1>{title}</h1>
              <p className="muted">Machine Learning & Explainable AI Academic Document Triage</p>
            </div>
            <button className="button secondary" onClick={refresh}>↻ <span>Refresh</span></button>
          </div>

          {notice && <div className="alert success">{notice}</div>}
          {error && <div className="alert error">{error}</div>}

          {active === "dashboard" && <Dashboard stats={stats} documents={documents} setActive={setActive} onOpen={openDocument} />}
          {active === "students" && admin?.role === "admin" && (
            <Students
              students={students}
              query={query}
              setQuery={setQuery}
              form={studentForm}
              setForm={setStudentForm}
              editingId={editingStudentId}
              onCancelEdit={() => { setEditingStudentId(null); setStudentForm({ full_name: "", email: "", registration_number: "" }); }}
              onEdit={editStudent}
              onDelete={deleteStudent}
              onSubmit={addStudent}
              busy={busy}
            />
          )}
          {active === "upload" && admin?.role === "admin" && (
            <Upload
              students={students}
              form={uploadForm}
              setForm={setUploadForm}
              onSubmit={uploadDocument}
              busy={busy}
            />
          )}
          {active === "documents" && (
            <DocumentList
              rows={filteredDocuments}
              query={query}
              setQuery={setQuery}
              statusFilter={statusFilter}
              setStatusFilter={setStatusFilter}
              onOpen={openDocument}
            />
          )}
          {active === "reviews" && (
            <ReviewQueue
              rows={filteredDocuments}
              query={query}
              setQuery={setQuery}
              statusFilter={statusFilter}
              setStatusFilter={setStatusFilter}
              selected={selected}
              onOpen={openDocument}
              onReview={review}
              onVerify={runVerification}
              onDownload={downloadDocument}
              busy={busy}
            />
          )}
          {active === "results" && (
            <Results
              rows={filteredDocuments}
              query={query}
              setQuery={setQuery}
              statusFilter={statusFilter}
              setStatusFilter={setStatusFilter}
              selected={selected}
              onOpen={openDocument}
              onVerify={runVerification}
              onDownload={downloadDocument}
              busy={busy}
            />
          )}
          {active === "model-lab" && <ModelLab summary={mlSummary} />}
          {active === "xai" && <ExplainableAI documents={documents} summary={mlSummary} selectedDoc={selected} onSelectDoc={openDocument} />}
          {active === "admin-users" && admin?.role === "admin" && (
            <AdminUsers users={adminUsers} onRefresh={() => api.get("/admin/users").then(({ data }) => setAdminUsers(data))} feedback={feedback} />
          )}
          {active === "history" && <History rows={documents} selected={selected} setSelected={setSelected} history={history} />}
          {active === "audit" && admin?.role === "admin" && <AuditLogView rows={auditLogs} />}
          {active === "settings" && <Settings health={health} />}
          {active === "profile" && <AdminProfile admin={admin} onLogout={logout} />}
        </div>
        <footer className="app-footer">Automated findings are decision-support indicators only; human review and institutional records are required for consequential decisions.</footer>
      </main>
    </div>
  );
}

function Dashboard({ stats, documents, setActive, onOpen }) {
  const cards = [
    ["Total students", stats?.total_students ?? "—", "♙", "blue"],
    ["Uploaded documents", stats?.total_documents ?? "—", "▤", "purple"],
    ["Verified records", stats?.verified_documents ?? "—", "✓", "green"],
    ["Awaiting review", stats?.pending_documents ?? "—", "◷", "amber"],
    ["Flagged alerts", stats?.suspicious_documents ?? "—", "!", "red"],
  ];
  return (
    <>
      <section className="stat-grid">
        {cards.map(([label, value, icon, color]) => (
          <article className="stat-card" key={label}>
            <div className={`stat-icon ${color}`}>{icon}</div>
            <div><span>{label}</span><strong>{value}</strong></div>
          </article>
        ))}
      </section>
      <section className="panel welcome-panel">
        <div>
          <p className="eyebrow">INTELLIGENT TRIAGE PIPELINE</p>
          <h2>Explainable AI & Machine Learning for Academic Records</h2>
          <p className="muted">
            Uploads undergo multi-layer verification: Tesseract OCR text extraction, structured grade parsing, mathematical validation, 5 classical ML models, DocumentCNN visual tampering detection, and local SHAP explanations.
          </p>
          <div style={{ display: "flex", gap: "10px", marginTop: "14px" }}>
            <button className="button primary" onClick={() => setActive("upload")}>Upload Document <span>→</span></button>
            <button className="button secondary" onClick={() => setActive("results")}>View Results <span>◉</span></button>
            <button className="button secondary" onClick={() => setActive("xai")}>Explore SHAP XAI <span>⚲</span></button>
          </div>
        </div>
        <div className="welcome-graphic"><div className="paper-icon">▤</div><div className="scan-line" /></div>
      </section>
      <section className="panel">
        <div className="section-title">
          <div><h2>Recent submissions</h2><p className="muted">Latest documents received for verification</p></div>
          <button className="text-button" onClick={() => setActive("documents")}>View all documents →</button>
        </div>
        <DocumentTable rows={documents.slice(0, 6)} onOpen={onOpen} />
      </section>
      <div className="disclaimer">
        <span>ⓘ</span>
        <p><strong>Ethical boundary:</strong> Automated anomaly alerts reflect image features and heuristic consistency rules. They do not constitute proof of document fraud or student misconduct without issuer verification.</p>
      </div>
    </>
  );
}

function Students({ students, query, setQuery, form, setForm, editingId, onCancelEdit, onEdit, onDelete, onSubmit, busy }) {
  const filteredStudents = students.filter((student) =>
    [student.full_name, student.registration_number, student.email]
      .some((value) => String(value || "").toLowerCase().includes(query.trim().toLowerCase())),
  );
  return (
    <section className="panel">
      <div className="section-title">
        <div>
          <h2>{editingId ? "Update student record" : "Register student"}</h2>
          <p className="muted">Manage student profiles for document matching and record cross-verification.</p>
        </div>
      </div>
      <form className="form-grid" onSubmit={onSubmit}>
        <label>Full name<input required minLength="2" maxLength="160" value={form.full_name} onChange={(e) => setForm({ ...form, full_name: e.target.value })} /></label>
        <label>Registration number<input maxLength="64" value={form.registration_number} onChange={(e) => setForm({ ...form, registration_number: e.target.value })} /></label>
        <label>Email address<input type="email" maxLength="254" value={form.email} onChange={(e) => setForm({ ...form, email: e.target.value })} /></label>
        <div className="form-action">
          <button className="button primary" disabled={busy}>{editingId ? "Save changes" : "Add student"}</button>
          {editingId && <button type="button" className="button secondary" onClick={onCancelEdit} style={{ marginLeft: "8px" }}>Cancel</button>}
        </div>
      </form>
      <div className="section-title table-heading">
        <div><h2>Student directory <span className="count-pill">{filteredStudents.length}</span></h2></div>
        <Search value={query} onChange={setQuery} />
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr><th>Student Name</th><th>Registration No.</th><th>Email</th><th>Created</th><th>Actions</th></tr>
          </thead>
          <tbody>
            {filteredStudents.map((student) => (
              <tr key={student.id}>
                <td><strong>{student.full_name}</strong><small>ID #{student.id}</small></td>
                <td>{student.registration_number || "—"}</td>
                <td>{student.email || "—"}</td>
                <td>{dateLabel(student.created_at)}</td>
                <td className="actions">
                  <button type="button" className="button tiny secondary" onClick={() => onEdit(student)}>Edit</button>
                  <button type="button" className="button tiny danger" onClick={() => onDelete(student)}>Delete</button>
                </td>
              </tr>
            ))}
            {!filteredStudents.length && <EmptyRow columns={5} text="No matching student records." />}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function Upload({ students, form, setForm, onSubmit, busy }) {
  const [isDragging, setIsDragging] = useState(false);
  const [preview, setPreview] = useState(null);

  function handleFileChange(file) {
    if (!file) {
      setPreview(null);
      setForm((cur) => ({ ...cur, file: null }));
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      alert("File exceeds maximum size limit of 10 MB.");
      return;
    }
    setForm((cur) => ({ ...cur, file }));
    if (file.type.startsWith("image/")) {
      const url = URL.createObjectURL(file);
      setPreview(url);
    } else {
      setPreview(null);
    }
  }

  return (
    <section className="panel upload-panel">
      <div className="section-title">
        <div>
          <h2>Upload academic document</h2>
          <p className="muted">Supported formats: PDF, JPG, JPEG, PNG · Maximum size: 10 MB</p>
        </div>
      </div>
      {!students.length && <div className="alert warning">Add a student record before uploading an academic document.</div>}
      <form className="stack-form" onSubmit={onSubmit}>
        <div className="form-grid">
          <label>
            Target student
            <select required value={form.student_id} onChange={(e) => setForm({ ...form, student_id: e.target.value })}>
              <option value="">Select a student</option>
              {students.map((student) => (
                <option key={student.id} value={student.id}>{student.full_name} {student.registration_number ? `(${student.registration_number})` : ""}</option>
              ))}
            </select>
          </label>
          <label>
            Document type
            <select value={form.document_type} onChange={(e) => setForm({ ...form, document_type: e.target.value })}>
              {documentTypes.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
            </select>
          </label>
        </div>

        <label
          className={`file-drop ${isDragging ? "drag-active" : ""}`}
          onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setIsDragging(false);
            if (e.dataTransfer.files?.[0]) handleFileChange(e.dataTransfer.files[0]);
          }}
        >
          <span className="upload-icon">↑</span>
          <strong>{form.file?.name || "Drag and drop document here, or click to browse"}</strong>
          <span>Accepts high-resolution PDF scans, JPG, or PNG images</span>
          <input
            type="file"
            accept=".pdf,.jpg,.jpeg,.png,application/pdf,image/jpeg,image/png"
            required
            onChange={(e) => handleFileChange(e.target.files?.[0] || null)}
          />
        </label>

        {preview && (
          <div className="doc-preview-box">
            <img src={preview} alt="Upload preview" className="doc-preview-img" />
          </div>
        )}

        <div className="privacy-note">
          <span>▣</span>
          <span>Files are sanitized, hashed via SHA-256 for cryptographic deduplication, and stored in isolated document storage.</span>
        </div>
        <button className="button primary" disabled={busy || !students.length || !form.file}>
          {busy ? "Running OCR & ML Analysis Pipeline…" : "Upload and Analyze Document"}
        </button>
      </form>
    </section>
  );
}

function DocumentList({ rows, query, setQuery, statusFilter, setStatusFilter, onOpen }) {
  return (
    <section className="panel">
      <div className="section-title">
        <div><h2>All documents</h2><p className="muted">Search across filenames, student records, and registration IDs</p></div>
        <div className="table-tools">
          <StatusFilter value={statusFilter} onChange={setStatusFilter} />
          <Search value={query} onChange={setQuery} />
        </div>
      </div>
      <DocumentTable rows={rows} onOpen={onOpen} />
    </section>
  );
}

function ReviewQueue({ rows, query, setQuery, statusFilter, setStatusFilter, selected, onOpen, onReview, onVerify, onDownload, busy }) {
  return (
    <div className="results-layout">
      <section className="panel">
        <div className="section-title">
          <div><h2>Review queue</h2><p className="muted">Flagged and pending documents requiring human sign-off</p></div>
          <Search value={query} onChange={setQuery} />
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr><th>Document</th><th>Student</th><th>Primary Indicator</th><th>Risk Level</th><th>Actions</th></tr>
            </thead>
            <tbody>
              {rows.map((item) => (
                <tr key={item.id}>
                  <td>
                    <button className="table-link" onClick={() => onOpen(item)}>{item.original_filename}</button>
                    <small>{item.document_type}</small>
                  </td>
                  <td>{item.student_name}<small>{item.student_registration_number || "No reg"}</small></td>
                  <td>{item.latest_result?.indicators?.[0] || "Review document and records"}</td>
                  <td><Risk value={item.latest_result?.risk_level} score={item.latest_result?.risk_score} /></td>
                  <td className="actions">
                    <button className="button tiny secondary" disabled={busy} onClick={() => onReview(item, "verified")}>Verify</button>
                    <button className="button tiny danger" disabled={busy} onClick={() => onReview(item, "suspicious")}>Flag</button>
                  </td>
                </tr>
              ))}
              {!rows.length && <EmptyRow columns={5} text="No pending documents in review queue." />}
            </tbody>
          </table>
        </div>
      </section>
      {selected && (
        <section className="panel detail-panel">
          <DocumentDetails item={selected} onVerify={onVerify} onDownload={onDownload} busy={busy} />
          <div className="review-actions">
            <button className="button primary" disabled={busy} onClick={() => onReview(selected, "verified")}>Approve & Verify</button>
            <button className="button danger" disabled={busy} onClick={() => onReview(selected, "suspicious")}>Flag as Suspicious</button>
          </div>
        </section>
      )}
    </div>
  );
}

function Results({ rows, query, setQuery, statusFilter, setStatusFilter, selected, onOpen, onVerify, onDownload, busy }) {
  return (
    <div className="results-layout">
      <section className="panel">
        <div className="section-title">
          <div><h2>Verification results</h2><p className="muted">Detailed fraud triage, academic field extraction, and SHAP explainability</p></div>
          <div className="table-tools">
            <StatusFilter value={statusFilter} onChange={setStatusFilter} />
            <Search value={query} onChange={setQuery} />
          </div>
        </div>
        <DocumentTable rows={rows} onOpen={onOpen} />
      </section>
      {selected && (
        <section className="panel detail-panel">
          <DocumentDetails item={selected} onVerify={onVerify} onDownload={onDownload} busy={busy} />
        </section>
      )}
    </div>
  );
}

function DocumentDetails({ item, onVerify, onDownload, busy }) {
  const [activeTab, setActiveTab] = useState("academic");
  const [localExplanation, setLocalExplanation] = useState(null);
  const [loadingExpl, setLoadingExpl] = useState(false);
  const [previewBlob, setPreviewBlob] = useState(null);

  const fields = item.extracted_fields || {};
  const validation = fields.validation || {};
  const subjects = fields.subjects || [];

  // Fetch local SHAP explanation
  useEffect(() => {
    let mounted = true;
    setLoadingExpl(true);
    api.get(`/documents/${item.id}/explanation`)
      .then(({ data }) => {
        if (mounted) setLocalExplanation(data.explanation);
      })
      .catch(() => {
        if (mounted) setLocalExplanation(null);
      })
      .finally(() => {
        if (mounted) setLoadingExpl(false);
      });
    return () => { mounted = false; };
  }, [item.id]);

  // Fetch preview image if png/jpg
  useEffect(() => {
    let mounted = true;
    if (item.media_type?.startsWith("image/")) {
      api.get(`/documents/${item.id}/file`, { responseType: "blob" })
        .then(({ data }) => {
          if (mounted) setPreviewBlob(URL.createObjectURL(data));
        })
        .catch(() => {
          if (mounted) setPreviewBlob(null);
        });
    } else {
      setPreviewBlob(null);
    }
    return () => {
      mounted = false;
    };
  }, [item.id, item.media_type]);

  return (
    <>
      <div className="section-title">
        <div>
          <p className="eyebrow">DOCUMENT #{item.id}</p>
          <h2>Verification Report</h2>
        </div>
        <StatusBadge value={item.status} />
      </div>

      <div className="detail-meta">
        <span>Student</span><strong>{item.student_name}</strong>
        <span>Filename</span><strong>{item.original_filename}</strong>
        <span>Doc Type</span><strong style={{ textTransform: "capitalize" }}>{item.document_type.replaceAll("_", " ")}</strong>
        <span>OCR Engine</span><strong>{item.ocr_confidence ? `Tesseract OCR (${item.ocr_confidence}% confidence)` : "Text Parser"}</strong>
      </div>

      <div className="risk-box">
        <div>
          <span>Composite Risk Score</span>
          <strong>{item.latest_result?.risk_score ?? "—"}<small>/100</small></strong>
        </div>
        <Risk value={item.latest_result?.risk_level} score={item.latest_result?.risk_score} />
      </div>

      {previewBlob && (
        <div className="doc-preview-box">
          <img src={previewBlob} alt="Document preview" className="doc-preview-img" />
        </div>
      )}

      {/* Tabs */}
      <div style={{ display: "flex", gap: "6px", margin: "14px 0 10px", borderBottom: "1px solid #e2e8f0", paddingBottom: "6px" }}>
        <button
          type="button"
          className={`button tiny ${activeTab === "academic" ? "primary" : "secondary"}`}
          onClick={() => setActiveTab("academic")}
        >
          Extracted Academics
        </button>
        <button
          type="button"
          className={`button tiny ${activeTab === "shap" ? "primary" : "secondary"}`}
          onClick={() => setActiveTab("shap")}
        >
          Local SHAP Explanation
        </button>
        <button
          type="button"
          className={`button tiny ${activeTab === "models" ? "primary" : "secondary"}`}
          onClick={() => setActiveTab("models")}
        >
          Model Predictions
        </button>
      </div>

      {activeTab === "academic" && (
        <div className="academic-card">
          <div className="academic-grid">
            <div><span>University</span><strong>{fields.university_name || "—"}</strong></div>
            <div><span>Institution</span><strong>{fields.institution_name || "—"}</strong></div>
            <div><span>Course / Degree</span><strong>{fields.course_name || "—"}</strong></div>
            <div><span>Department</span><strong>{fields.department || "—"}</strong></div>
            <div><span>Semester</span><strong>{fields.semester || "—"}</strong></div>
            <div><span>Academic Year</span><strong>{fields.academic_year || "—"}</strong></div>
            <div><span>CGPA</span><strong>{fields.cgpa || (validation.calculated_cgpa ? `${validation.calculated_cgpa} (calculated)` : "—")}</strong></div>
            <div><span>Percentage</span><strong>{fields.percentage || (validation.calculated_percentage ? `${validation.calculated_percentage}% (calculated)` : "—")}</strong></div>
            <div><span>Issue Date</span><strong>{fields.issue_date || "—"}</strong></div>
            <div><span>Result Status</span><strong>{fields.result_status || "—"}</strong></div>
          </div>

          <div>
            <span>Academic Validation Status:</span>
            {validation.is_marks_consistent === true && <span className="badge-clean">✓ Marks & Totals Consistent</span>}
            {validation.is_marks_consistent === false && <span className="badge-flag">⚠ Inconsistent Marks / Anomalies Detected</span>}
          </div>

          {validation.anomalies_detected?.length > 0 && (
            <div style={{ marginTop: "8px" }}>
              {validation.anomalies_detected.map((msg, idx) => (
                <div key={idx} className="alert error" style={{ fontSize: "10px", padding: "6px 8px", marginBottom: "4px" }}>
                  {msg}
                </div>
              ))}
            </div>
          )}

          {subjects.length > 0 && (
            <table className="academic-table">
              <thead>
                <tr>
                  <th>Code</th>
                  <th>Subject Title</th>
                  <th>Max</th>
                  <th>Obtained</th>
                  <th>Grade</th>
                </tr>
              </thead>
              <tbody>
                {subjects.map((s, idx) => (
                  <tr key={idx}>
                    <td>{s.code || "—"}</td>
                    <td><strong>{s.name}</strong></td>
                    <td>{s.max_marks ?? "—"}</td>
                    <td>{s.obtained_marks ?? "—"}</td>
                    <td><span className="count-pill">{s.grade || "—"}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}

      {activeTab === "shap" && (
        <div className="shap-local-card">
          <h4>Local Explainable AI (SHAP Permutation Attribution)</h4>
          {loadingExpl && <p className="muted">Computing local SHAP explanations…</p>}
          {localExplanation?.status === "available" && (
            <>
              <p style={{ fontSize: "11px", color: "#334155", lineHeight: "1.5" }}>
                <strong>Model Reasoning:</strong> {localExplanation.narrative_explanation}
              </p>
              {localExplanation.plot_base64 && (
                <img src={localExplanation.plot_base64} alt="Local SHAP plot" className="shap-local-plot" />
              )}
              <div style={{ marginTop: "12px" }}>
                {localExplanation.feature_contributions?.map((fc) => (
                  <div key={fc.feature} className="shap-contrib-item">
                    <div>
                      <span><strong>{fc.friendly_name}</strong>: {fc.value}</span>
                      <small style={{ display: "block", color: "#64748b", fontSize: "9px" }}>{fc.description}</small>
                    </div>
                    <span className={`shap-contrib-badge ${fc.direction}`}>
                      {fc.shap_value > 0 ? `+${fc.shap_value}` : fc.shap_value} ({fc.direction.replace("_", " ")})
                    </span>
                  </div>
                ))}
              </div>
            </>
          )}
          {localExplanation?.status !== "available" && !loadingExpl && (
            <p className="muted">Local SHAP explanation is not available for this record.</p>
          )}
        </div>
      )}

      {activeTab === "models" && (
        <div>
          <h4>Rule & Model Risk Indicators</h4>
          <ul className="indicator-list">
            {(item.latest_result?.indicators || item.processing_notes || []).map((line, index) => (
              <li key={index}>{line}</li>
            ))}
          </ul>
          <p className="model-note">{item.latest_result?.model_notes}</p>
        </div>
      )}

      <div className="detail-buttons" style={{ marginTop: "14px" }}>
        <button className="button secondary" onClick={() => onDownload(item)}>↓ Download Document</button>
        <button className="button secondary" disabled={busy} onClick={() => onVerify(item)}>↻ Re-run Analysis</button>
      </div>

      {item.extracted_text && (
        <details className="ocr-details">
          <summary>View raw extracted text ({item.extracted_text.length} characters)</summary>
          <pre>{item.extracted_text}</pre>
        </details>
      )}
    </>
  );
}

function ExplainableAI({ documents, summary, selectedDoc, onSelectDoc }) {
  const [docExpl, setDocExpl] = useState(null);
  const [selectedId, setSelectedId] = useState(selectedDoc?.id || (documents[0]?.id ?? ""));
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!selectedId) return;
    setLoading(true);
    api.get(`/documents/${selectedId}/explanation`)
      .then(({ data }) => setDocExpl(data.explanation))
      .catch(() => setDocExpl(null))
      .finally(() => setLoading(false));
  }, [selectedId]);

  const shapFeatures = summary?.shap?.features || [];
  const largest = Math.max(...shapFeatures.map((f) => f.mean_absolute_shap), 0);

  return (
    <div>
      <section className="panel">
        <div className="section-title">
          <div>
            <p className="eyebrow">TRANSPARENT & AUDITABLE DECISION SUPPORT</p>
            <h2>Explainable AI (SHAP) Overview</h2>
            <p className="muted">
              SHAP (SHapley Additive exPlanations) attributes the contribution of each extracted image and structural feature to the model's tampering probability score.
            </p>
          </div>
        </div>

        <h3>Global Feature Importances</h3>
        <p className="muted">Mean absolute Shapley values across held-out benchmark test documents.</p>
        {shapFeatures.length > 0 ? (
          <div className="shap-bars">
            {shapFeatures.map((f) => (
              <div className="shap-row" key={f.name}>
                <span>{f.name.replaceAll("_", " ")}</span>
                <div className="shap-track">
                  <i style={{ width: `${Math.max(2, (f.mean_absolute_shap / (largest || 1)) * 100)}%` }} />
                </div>
                <strong>{f.mean_absolute_shap.toPrecision(3)}</strong>
              </div>
            ))}
          </div>
        ) : (
          <p className="muted">No global SHAP artifacts available. Run model training first.</p>
        )}
      </section>

      <section className="panel">
        <div className="section-title">
          <div>
            <h2>Document-Level Local Explanation</h2>
            <p className="muted">Inspect how specific features contributed to an individual document's risk assessment.</p>
          </div>
          <select
            className="history-select"
            value={selectedId}
            onChange={(e) => {
              const id = Number(e.target.value);
              setSelectedId(id);
              const found = documents.find((d) => d.id === id);
              if (found) onSelectDoc(found);
            }}
          >
            <option value="">Select document to explain</option>
            {documents.map((d) => (
              <option key={d.id} value={d.id}>Doc #{d.id}: {d.original_filename} ({d.student_name})</option>
            ))}
          </select>
        </div>

        {loading && <p className="muted">Calculating local SHAP feature attributions…</p>}

        {docExpl && docExpl.status === "available" && (
          <div className="shap-local-card">
            <h4>Predicted Tampered Probability: {(docExpl.predicted_probability * 100).toFixed(1)}% ({docExpl.prediction.replace("_", " ")})</h4>
            <p style={{ fontSize: "12px", color: "#334155" }}>
              <strong>Natural Language Summary:</strong> {docExpl.narrative_explanation}
            </p>
            {docExpl.plot_base64 && (
              <div style={{ textAlign: "center", margin: "14px 0" }}>
                <img src={docExpl.plot_base64} alt="Local SHAP waterfall chart" style={{ maxWidth: "100%", borderRadius: "8px" }} />
              </div>
            )}
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Feature Name</th>
                    <th>Document Value</th>
                    <th>SHAP Contribution</th>
                    <th>Risk Impact</th>
                  </tr>
                </thead>
                <tbody>
                  {docExpl.feature_contributions?.map((fc) => (
                    <tr key={fc.feature}>
                      <td><strong>{fc.friendly_name}</strong><small>{fc.description}</small></td>
                      <td>{fc.value}</td>
                      <td><strong>{fc.shap_value > 0 ? `+${fc.shap_value}` : fc.shap_value}</strong></td>
                      <td>
                        <span className={`shap-contrib-badge ${fc.direction}`}>
                          {fc.direction === "increases_risk" ? "Elevates Risk" : "Reduces Risk"}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </section>
    </div>
  );
}

function AdminUsers({ users, onRefresh, feedback }) {
  const [form, setForm] = useState({ username: "", password: "", role: "admin" });
  const [busy, setBusy] = useState(false);

  async function submit(e) {
    e.preventDefault();
    if (form.password.length < 12) {
      alert("Password must be at least 12 characters.");
      return;
    }
    setBusy(true);
    try {
      await api.post("/admin/users", form);
      feedback(true, `User '${form.username}' created successfully.`);
      setForm({ username: "", password: "", role: "admin" });
      onRefresh();
    } catch (err) {
      feedback(false, messageFrom(err));
    } finally {
      setBusy(false);
    }
  }

  async function deleteUser(u) {
    if (!window.confirm(`Delete administrator ${u.username}?`)) return;
    try {
      await api.delete(`/admin/users/${u.id}`);
      feedback(true, `User '${u.username}' removed.`);
      onRefresh();
    } catch (err) {
      feedback(false, messageFrom(err));
    }
  }

  return (
    <section className="panel">
      <div className="section-title">
        <div>
          <h2>Administrator Management</h2>
          <p className="muted">Manage staff accounts with role-based access control (Admin or Reviewer)</p>
        </div>
      </div>

      <form className="form-grid" onSubmit={submit}>
        <label>
          Username
          <input required minLength="3" maxLength="64" value={form.username} onChange={(e) => setForm({ ...form, username: e.target.value })} />
        </label>
        <label>
          Password (min 12 chars)
          <input type="password" required minLength="12" value={form.password} onChange={(e) => setForm({ ...form, password: e.target.value })} />
        </label>
        <label>
          Role
          <select value={form.role} onChange={(e) => setForm({ ...form, role: e.target.value })}>
            <option value="admin">Administrator (Full CRUD & User Management)</option>
            <option value="reviewer">Reviewer (Read & Document Review Sign-off)</option>
          </select>
        </label>
        <div className="form-action">
          <button className="button primary" disabled={busy}>{busy ? "Creating User…" : "Add Staff Account"}</button>
        </div>
      </form>

      <div className="section-title table-heading">
        <div><h2>Registered Staff Accounts <span className="count-pill">{users.length}</span></h2></div>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr><th>Account</th><th>Role</th><th>Active</th><th>Created Date</th><th>Actions</th></tr>
          </thead>
          <tbody>
            {users.map((u) => (
              <tr key={u.id}>
                <td><strong>{u.username}</strong><small>ID #{u.id}</small></td>
                <td><span className="count-pill">{u.role}</span></td>
                <td>{u.is_active ? "Active" : "Disabled"}</td>
                <td>{dateLabel(u.created_at)}</td>
                <td className="actions">
                  <button type="button" className="button tiny danger" onClick={() => deleteUser(u)}>Delete</button>
                </td>
              </tr>
            ))}
            {!users.length && <EmptyRow columns={5} text="No staff accounts found." />}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function ModelLab({ summary }) {
  if (!summary) return <section className="panel"><p className="muted">Loading persisted model results…</p></section>;
  if (summary.status === "dataset_required") {
    return (
      <section className="panel model-lab">
        <p className="eyebrow">RESEARCH MODELS</p>
        <h2>Training results are not available yet</h2>
        <p>{summary.message}</p>
      </section>
    );
  }
  const algorithms = Object.entries(summary.models || {});
  const shapFeatures = summary.shap?.features || [];
  const largestContribution = Math.max(...shapFeatures.map((item) => item.mean_absolute_shap), 0);
  return (
    <div className="model-lab">
      <section className="panel">
        <div className="section-title">
          <div>
            <p className="eyebrow">HELD-OUT EVALUATION</p>
            <h2>Classical ML Algorithm Comparison</h2>
            <p className="muted">Five algorithms evaluated on held-out test groups with zero data leakage.</p>
          </div>
          <StatusBadge value={summary.status} />
        </div>
        {summary.row_counts && (
          <p className="muted">
            Total Rows: {summary.row_counts.total} · Train: {summary.row_counts.train} · Validation: {summary.row_counts.validation} · Held-Out Test: {summary.row_counts.test} · Group leakage check: passed
          </p>
        )}
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Algorithm</th>
                <th>Accuracy</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1 Score</th>
                <th>ROC-AUC</th>
                <th>Prediction ms</th>
              </tr>
            </thead>
            <tbody>
              {algorithms.map(([name, metrics]) => (
                <tr key={name}>
                  <td>
                    <strong>{name.replaceAll("_", " ")}</strong>
                    {summary.selected_supervised_model === name && <small style={{ color: "#2563eb", fontWeight: "bold" }}>Best Supervised Model</small>}
                  </td>
                  <td>{formatMetric(metrics.accuracy)}</td>
                  <td>{formatMetric(metrics.precision)}</td>
                  <td>{formatMetric(metrics.recall)}</td>
                  <td>{formatMetric(metrics.f1)}</td>
                  <td>{formatMetric(metrics.roc_auc)}</td>
                  <td>{formatMetric(metrics.prediction_ms_per_item)}</td>
                </tr>
              ))}
              {!algorithms.length && <EmptyRow columns={7} text="No classical model evaluation recorded." />}
            </tbody>
          </table>
        </div>
      </section>

      <section className="panel">
        <div className="section-title">
          <div>
            <p className="eyebrow">DEEP LEARNING MODEL</p>
            <h2>DocumentCNN Deep Learning Architecture</h2>
          </div>
        </div>
        {summary.cnn ? (
          <div>
            <p style={{ fontSize: "12px", color: "#334155" }}>
              <strong>Architecture:</strong> {summary.cnn.architecture} (Image size: 224x224 RGB, Batch Normalization, Dropout 0.35, Adaptive Pooling)
            </p>
            <div className="stat-grid" style={{ marginTop: "12px" }}>
              <div className="stat-card">
                <div><span>CNN Test Accuracy</span><strong>{formatMetric(summary.cnn.metrics?.accuracy)}</strong></div>
              </div>
              <div className="stat-card">
                <div><span>CNN Test Recall</span><strong>{formatMetric(summary.cnn.metrics?.recall)}</strong></div>
              </div>
              <div className="stat-card">
                <div><span>CNN Test F1</span><strong>{formatMetric(summary.cnn.metrics?.f1)}</strong></div>
              </div>
              <div className="stat-card">
                <div><span>Selected Epoch</span><strong>{summary.cnn.metrics?.selected_epoch || "—"}</strong></div>
              </div>
            </div>
            <p className="muted" style={{ marginTop: "10px" }}>
              Honest Disclosure: Without ImageNet pretraining, DocumentCNN trained from scratch reflects dataset baseline patterns. Supervised tabular models provide the primary high-confidence inference.
            </p>
          </div>
        ) : (
          <p className="muted">CNN training artifacts are unavailable.</p>
        )}
      </section>

      <section className="panel">
        <div className="section-title">
          <div>
            <p className="eyebrow">EXPLAINABLE AI</p>
            <h2>Global SHAP Feature Attribution</h2>
          </div>
        </div>
        {shapFeatures.length ? (
          <div className="shap-bars">
            {shapFeatures.map((feature) => (
              <div className="shap-row" key={feature.name}>
                <span>{feature.name.replaceAll("_", " ")}</span>
                <div className="shap-track">
                  <i style={{ width: `${Math.max(1, (feature.mean_absolute_shap / largestContribution) * 100)}%` }} />
                </div>
                <strong>{feature.mean_absolute_shap.toPrecision(3)}</strong>
              </div>
            ))}
          </div>
        ) : (
          <p className="muted">No SHAP explanation artifact is available yet.</p>
        )}
      </section>

      <section className="disclaimer">
        <span>ⓘ</span>
        <p>{(summary.limitations || []).join(" ")}</p>
      </section>
    </div>
  );
}

function Settings({ health }) {
  return (
    <section className="panel">
      <div className="section-title">
        <div>
          <h2>System Configuration & Diagnostics</h2>
          <p className="muted">Runtime environment, database connection, OCR engine, and security parameters</p>
        </div>
      </div>

      <div className="settings-grid">
        <div className="settings-card">
          <strong>Database Engine</strong>
          <span>Connection Status</span>
          <p style={{ color: "#16a34a" }}>● Connected ({health?.database || "Active"})</p>
          <span>Architecture</span>
          <p>SQLite (Development Default) & PostgreSQL 18 Compatible via SQLAlchemy</p>
          <span>Migrations</span>
          <p>Alembic Version 20261114_0001 (head)</p>
        </div>

        <div className="settings-card">
          <strong>OCR Engine (Tesseract)</strong>
          <span>Availability</span>
          <p style={{ color: health?.ocr?.available ? "#16a34a" : "#dc2626" }}>
            ● {health?.ocr?.available ? "Online & Detected" : "Offline / Path Unconfigured"}
          </p>
          <span>Version</span>
          <p>{health?.ocr?.version || "5.5.3"}</p>
          <span>Executable Path</span>
          <p style={{ fontSize: "11px" }}>{health?.ocr?.executable || "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"}</p>
        </div>

        <div className="settings-card">
          <strong>Security & Authentication</strong>
          <span>Password Hashing</span>
          <p>Argon2id (Cryptographically salted, memory-hard parameters)</p>
          <span>Session Tokens</span>
          <p>JSON Web Tokens (JWT) with HMAC-SHA256 signature, 8-hour expiry</p>
          <span>Brute Force Protection</span>
          <p>In-memory sliding window rate limiting (5 attempts per 5 minutes)</p>
        </div>

        <div className="settings-card">
          <strong>Storage & Upload Policies</strong>
          <span>File Size Cap</span>
          <p>10 MB per document</p>
          <span>Allowed File Types</span>
          <p>PDF (PyMuPDF render), JPEG, PNG (Pillow & OpenCV validation)</p>
          <span>Integrity Verification</span>
          <p>Cryptographic SHA-256 duplicate rejection and hash verification</p>
        </div>
      </div>
    </section>
  );
}

function History({ rows, selected, setSelected, history }) {
  return (
    <section className="panel">
      <div className="section-title">
        <div><h2>Document audit history</h2><p className="muted">Chronological timeline of upload, analysis, and administrator review events</p></div>
        <select className="history-select" value={selected?.id || ""} onChange={(e) => setSelected(rows.find((item) => item.id === Number(e.target.value)) || null)}>
          <option value="">Select a document</option>
          {rows.map((item) => <option key={item.id} value={item.id}>Doc #{item.id}: {item.original_filename}</option>)}
        </select>
      </div>
      {selected && (
        <div className="timeline">
          {history.map((event) => (
            <div className="timeline-item" key={event.id}>
              <span className="timeline-dot" />
              <div>
                <strong>{event.action.replaceAll("_", " ")}</strong>
                <time>{dateLabel(event.created_at)}</time>
                <pre>{JSON.stringify(event.details, null, 2)}</pre>
              </div>
            </div>
          ))}
          {!history.length && <p className="muted">No history entries recorded for this document.</p>}
        </div>
      )}
      {!rows.length && <div className="empty-state">No documents have been uploaded.</div>}
    </section>
  );
}

function DocumentTable({ rows, onOpen }) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Document</th>
            <th>Student</th>
            <th>Uploaded</th>
            <th>Risk Assessment</th>
            <th>Review Status</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((item) => (
            <tr key={item.id}>
              <td>
                <button className="table-link" onClick={() => onOpen(item)}>{item.original_filename}</button>
                <small>{item.document_type.replaceAll("_", " ")} · {(item.file_size / 1024).toFixed(0)} KB</small>
              </td>
              <td>{item.student_name}<small>{item.student_registration_number || "No registration no."}</small></td>
              <td>{dateLabel(item.created_at)}</td>
              <td><Risk value={item.latest_result?.risk_level} score={item.latest_result?.risk_score} /></td>
              <td><StatusBadge value={item.status} /></td>
            </tr>
          ))}
          {!rows.length && <EmptyRow columns={5} text="No documents found." />}
        </tbody>
      </table>
    </div>
  );
}

function AdminProfile({ admin, onLogout }) {
  return (
    <section className="panel profile-panel">
      <div className="section-title">
        <div><h2>Administrator profile</h2><p className="muted">Authenticated backend session credentials</p></div>
      </div>
      <div className="profile-card">
        <span className="profile-avatar">{(admin?.username || "A").slice(0, 1).toUpperCase()}</span>
        <div>
          <strong>{admin?.username || "Staff account"}</strong>
          <span>{admin?.role || "staff"}</span>
          <small>Signed in with Argon2id salted credentials.</small>
        </div>
      </div>
      <div className="privacy-note">
        <span>▣</span>
        <span>Signing out invalidates your browser token. Always sign out after administrative work on shared terminals.</span>
      </div>
      <button className="button secondary" onClick={onLogout} style={{ marginTop: "14px" }}>Sign out</button>
    </section>
  );
}

function AuditLogView({ rows }) {
  return (
    <section className="panel">
      <div className="section-title">
        <div><h2>Security audit log</h2><p className="muted">Immutable record of authentication, document intake, and review decisions</p></div>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr><th>Event Action</th><th>Account</th><th>Target Resource</th><th>Client IP</th><th>Timestamp</th></tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.id}>
                <td><strong>{row.event_type.replaceAll("_", " ")}</strong></td>
                <td>{row.admin_id ? `Admin #${row.admin_id}` : "Unauthenticated"}</td>
                <td>{row.target_type ? `${row.target_type} #${row.target_id || ""}` : "—"}</td>
                <td>{row.ip_address || "127.0.0.1"}</td>
                <td>{dateLabel(row.created_at)}</td>
              </tr>
            ))}
            {!rows.length && <EmptyRow columns={5} text="No audit events recorded." />}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export default App;

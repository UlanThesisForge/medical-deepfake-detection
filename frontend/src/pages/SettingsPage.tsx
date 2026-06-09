// SettingsPage.tsx
import { useState } from "react";
import { useAuth } from "../App";
import { api } from "../api/client";

export default function SettingsPage() {
  const { user } = useAuth();
  const [name, setName] = useState(user?.full_name || "");
  const [pass, setPass] = useState("");
  const [pass2, setPass2] = useState("");
  const [msg, setMsg] = useState("");
  const [loading, setLoading] = useState(false);

  const save = async () => {
    if (pass && pass !== pass2) {
      setMsg("Пароли не совпадают");
      return;
    }
    setLoading(true);
    setMsg("");
    try {
      const body: Record<string, string> = {};
      if (name !== user?.full_name) body.full_name = name;
      if (pass) body.password = pass;
      await api.patch(`/auth/me`, body);
      setMsg("Сохранено");
      setPass("");
      setPass2("");
    } catch (e: unknown) {
      setMsg(e instanceof Error ? e.message : "Ошибка");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fade-in" style={{ maxWidth: 600 }}>
      <div className="page-header">
        <div>
          <h1 className="page-title">Настройки</h1>
          <p className="page-subtitle">Управление профилем</p>
        </div>
      </div>

      <div className="card">
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 16,
            marginBottom: 20,
          }}
        >
          <div
            style={{
              width: 52,
              height: 52,
              borderRadius: "50%",
              background: "var(--accent-dim)",
              color: "var(--accent-light)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: "1.4rem",
              fontWeight: 600,
            }}
          >
            {user?.full_name?.charAt(0).toUpperCase()}
          </div>
          <div>
            <div style={{ fontWeight: 500, fontSize: "1rem" }}>
              {user?.full_name}
            </div>
            <span className={`badge-${user?.role}`} style={{ marginTop: 4 }}>
              {user?.role}
            </span>
            {user?.organization && (
              <div
                style={{
                  fontSize: ".8rem",
                  color: "var(--text-muted)",
                  marginTop: 2,
                }}
              >
                {user.organization}
              </div>
            )}
          </div>
        </div>

        <div className="divider" />

        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <div className="form-group">
            <label className="label">Полное имя</label>
            <input
              className="input"
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
          </div>
          <div className="form-group">
            <label className="label">Email</label>
            <input
              className="input"
              value={user?.email || ""}
              disabled
              style={{ opacity: 0.6 }}
            />
          </div>
          <div className="form-group">
            <label className="label">Новый пароль</label>
            <input
              className="input"
              type="password"
              placeholder="Оставьте пустым если не меняете"
              value={pass}
              onChange={(e) => setPass(e.target.value)}
            />
          </div>
          <div className="form-group">
            <label className="label">Подтвердите пароль</label>
            <input
              className="input"
              type="password"
              placeholder="••••••••"
              value={pass2}
              onChange={(e) => setPass2(e.target.value)}
            />
          </div>

          {msg && (
            <div
              style={{
                padding: "10px 14px",
                borderRadius: "var(--radius-md)",
                fontSize: ".875rem",
                background:
                  msg === "Сохранено"
                    ? "rgba(34,197,94,.1)"
                    : "var(--danger-dim)",
                color: msg === "Сохранено" ? "#4ade80" : "#f87171",
                border: `1px solid ${msg === "Сохранено" ? "rgba(34,197,94,.25)" : "rgba(239,68,68,.25)"}`,
              }}
            >
              {msg}
            </div>
          )}

          <button
            className="btn btn-primary"
            onClick={save}
            disabled={loading}
            style={{ alignSelf: "flex-start" }}
          >
            {loading ? "Сохраняю..." : "Сохранить изменения"}
          </button>
        </div>
      </div>
    </div>
  );
}

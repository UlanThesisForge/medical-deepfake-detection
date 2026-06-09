import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../App";
import { api, AuthResponse } from "../api/client";
import "./Login.css";

export default function Login() {
  const { login, user } = useAuth();
  const navigate = useNavigate();

  const [tab, setTab] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [org, setOrg] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  // Все хуки до return
  useEffect(() => {
    if (user) navigate("/analysis", { replace: true });
  }, [user, navigate]);

  if (user) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      let data: AuthResponse;
      if (tab === "login") {
        data = await api.post<AuthResponse>("/auth/login", { email, password });
      } else {
        data = await api.post<AuthResponse>("/auth/register", {
          full_name: name,
          email,
          password,
          organization: org,
        });
      }
      login(data);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Ошибка");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="login-page">
      <div className="login-bg">
        <div className="login-grid" />
      </div>

      <div className="login-left">
        <div className="login-brand">
          <div className="login-logo">
            <svg
              width="28"
              height="28"
              viewBox="0 0 24 24"
              fill="none"
              stroke="white"
              strokeWidth="2"
            >
              <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
              <path d="M9 12l2 2 4-4" />
            </svg>
          </div>
          <span className="login-brand-name">DeepfakeMedical</span>
        </div>

        <div className="login-tagline">
          <h1>
            Обнаружение
            <br />
            дипфейков в<br />
            медицинских снимках
          </h1>
          <p>
            Система машинного обучения для выявления синтетически
            сгенерированных медицинских изображений в делах о мошенничестве в
            сфере здравоохранения.
          </p>
        </div>

        <div className="login-stats">
          {[
            { val: "AUC 0.961", lbl: "Точность" },
            { val: "91.4%", lbl: "Accuracy" },
            { val: "3 мод.", lbl: "Detectie" },
          ].map((s) => (
            <div key={s.lbl} className="login-stat">
              <span className="stat-val">{s.val}</span>
              <span className="stat-lbl">{s.lbl}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="login-right">
        <div className="login-card fade-in">
          <div className="login-tabs">
            {(["login", "register"] as const).map((t) => (
              <button
                key={t}
                className={`login-tab ${tab === t ? "active" : ""}`}
                onClick={() => {
                  setTab(t);
                  setError("");
                }}
              >
                {t === "login" ? "Вход" : "Регистрация"}
              </button>
            ))}
          </div>

          <form onSubmit={handleSubmit} className="login-form">
            {tab === "register" && (
              <>
                <div className="form-group">
                  <label className="label">Полное имя</label>
                  <input
                    className="input"
                    placeholder="Иванов Иван Иванович"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    required
                  />
                </div>
                <div className="form-group">
                  <label className="label">Организация</label>
                  <input
                    className="input"
                    placeholder="Insurance Investigation Dept."
                    value={org}
                    onChange={(e) => setOrg(e.target.value)}
                  />
                </div>
              </>
            )}
            <div className="form-group">
              <label className="label">Email</label>
              <input
                className="input"
                type="email"
                placeholder="analyst@forensics.kz"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
              />
            </div>
            <div className="form-group">
              <label className="label">Пароль</label>
              <input
                className="input"
                type="password"
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                minLength={6}
              />
            </div>

            {error && (
              <div className="login-error">
                <svg
                  width="13"
                  height="13"
                  viewBox="0 0 24 24"
                  fill="currentColor"
                >
                  <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z" />
                </svg>
                {error}
              </div>
            )}

            <button
              type="submit"
              className="btn btn-primary login-submit"
              disabled={loading}
            >
              {loading ? (
                <>
                  <span className="spinner" style={{ width: 15, height: 15 }} />{" "}
                  Подождите...
                </>
              ) : tab === "login" ? (
                "Войти"
              ) : (
                "Создать аккаунт"
              )}
            </button>
          </form>

          {tab === "login" && (
            <p className="login-hint">
              Демо: <span className="mono">admin@deepfake-medical.kz</span> /{" "}
              <span className="mono">admin123</span>
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

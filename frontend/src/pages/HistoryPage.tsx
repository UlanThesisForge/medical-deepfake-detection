// HistoryPage.tsx
import { useState, useEffect } from "react";
import { api, HistoryItem } from "../api/client";

export default function HistoryPage() {
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState<string | null>(null);

  useEffect(() => {
    api
      .get<HistoryItem[]>("/api/v1/images/history?limit=50")
      .then((d) => setHistory(d))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="fade-in" style={{ maxWidth: 900 }}>
      <div className="page-header">
        <div>
          <h1 className="page-title">История анализов</h1>
          <p className="page-subtitle">Все проверенные изображения</p>
        </div>
        <span style={{ fontSize: ".85rem", color: "var(--text-muted)" }}>
          {history.length} записей
        </span>
      </div>

      {loading ? (
        <div style={{ display: "flex", justifyContent: "center", padding: 60 }}>
          <div className="spinner" style={{ width: 32, height: 32 }} />
        </div>
      ) : history.length === 0 ? (
        <div
          className="card"
          style={{
            textAlign: "center",
            padding: 60,
            color: "var(--text-muted)",
          }}
        >
          <p style={{ fontSize: "2rem", marginBottom: 12 }}>📋</p>
          <p>История пуста. Выполните первый анализ!</p>
        </div>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          {history.map((item) => {
            const isDF = item.label === "deepfake";
            const color = isDF ? "var(--deepfake)" : "var(--authentic)";
            const open = selected === item.result_id;

            return (
              <div
                key={item.result_id}
                className="card"
                style={{
                  cursor: "pointer",
                  borderColor: open ? color : undefined,
                  transition: "border-color .15s",
                }}
                onClick={() => setSelected(open ? null : item.result_id)}
              >
                <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
                  <img
                    src={`http://localhost:8000${item.image_url}`}
                    alt=""
                    style={{
                      width: 52,
                      height: 52,
                      borderRadius: 8,
                      objectFit: "cover",
                      background: "var(--bg-elevated)",
                      flexShrink: 0,
                    }}
                    onError={(e) => {
                      (e.target as HTMLImageElement).style.display = "none";
                    }}
                  />
                  <div style={{ flex: 1 }}>
                    <div
                      style={{
                        fontWeight: 500,
                        color,
                        fontFamily: "var(--font-mono)",
                        fontSize: ".9rem",
                      }}
                    >
                      {item.label.toUpperCase()}
                    </div>
                    <div
                      style={{
                        fontSize: ".78rem",
                        color: "var(--text-muted)",
                        marginTop: 2,
                      }}
                    >
                      {item.filename} ·{" "}
                      {new Date(item.created_at).toLocaleString("ru-RU")}
                    </div>
                  </div>
                  <span className={`badge-${item.label}`}>
                    {(item.confidence * 100).toFixed(0)}%
                  </span>
                  <svg
                    width="14"
                    height="14"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="var(--text-muted)"
                    strokeWidth="2"
                    style={{
                      transform: open ? "rotate(180deg)" : "rotate(0)",
                      transition: "transform .2s",
                      flexShrink: 0,
                    }}
                  >
                    <path d="M6 9l6 6 6-6" />
                  </svg>
                </div>

                {open && (
                  <div
                    style={{
                      marginTop: 14,
                      paddingTop: 14,
                      borderTop: "1px solid var(--border-light)",
                      display: "flex",
                      gap: 16,
                    }}
                  >
                    <img
                      src={`http://localhost:8000${item.heatmap_url}`}
                      alt="Grad-CAM"
                      style={{
                        width: 160,
                        height: 160,
                        borderRadius: 8,
                        objectFit: "cover",
                        background: "var(--bg-elevated)",
                      }}
                    />
                    <div
                      style={{
                        fontSize: ".85rem",
                        color: "var(--text-secondary)",
                      }}
                    >
                      <p
                        style={{
                          marginBottom: 8,
                          fontWeight: 500,
                          color: "var(--text-primary)",
                        }}
                      >
                        Grad-CAM — зоны артефактов
                      </p>
                      <p>
                        Время обработки:{" "}
                        <span className="mono">{item.processing_ms} мс</span>
                      </p>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

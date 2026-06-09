// StatisticsPage.tsx
import { useState, useEffect } from "react";
import { api, StatsResponse } from "../api/client";

export default function StatisticsPage() {
  const [stats, setStats] = useState<StatsResponse | null>(null);
  useEffect(() => {
    api
      .get<StatsResponse>("/api/v1/stats")
      .then(setStats)
      .catch(() => {});
  }, []);
  const total = stats?.total || 0;
  const byLabel = stats?.by_label || {};

  return (
    <div className="fade-in">
      <div className="page-header">
        <div>
          <h1 className="page-title">Статистика</h1>
          <p className="page-subtitle">Сводка по выполненным анализам</p>
        </div>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill,minmax(180px,1fr))",
          gap: 14,
          marginBottom: 20,
        }}
      >
        {[
          { val: total, lbl: "Всего анализов" },
          { val: `${stats?.avg_processing_ms || 0} мс`, lbl: "Среднее время" },
          { val: stats?.model?.version || "—", lbl: "Версия модели" },
          {
            val: stats?.model
              ? `${((stats.model.auc_roc || 0) * 100).toFixed(1)}%`
              : "—",
            lbl: "AUC-ROC модели",
          },
        ].map((m, i) => (
          <div key={i} className="card" style={{ textAlign: "center" }}>
            <div
              style={{
                fontSize: "1.8rem",
                fontWeight: 700,
                fontFamily: "var(--font-mono)",
                color: "var(--accent-light)",
              }}
            >
              {m.val}
            </div>
            <div
              style={{
                fontSize: ".8rem",
                color: "var(--text-secondary)",
                marginTop: 4,
              }}
            >
              {m.lbl}
            </div>
          </div>
        ))}
      </div>

      {total > 0 && (
        <div className="card">
          <h3 style={{ marginBottom: 16 }}>Распределение результатов</h3>
          {Object.entries(byLabel).map(([label, count]) => {
            const pct = total > 0 ? (count / total) * 100 : 0;
            const color =
              label === "deepfake" ? "var(--deepfake)" : "var(--authentic)";
            return (
              <div
                key={label}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 12,
                  marginBottom: 12,
                  fontSize: ".875rem",
                }}
              >
                <span
                  className={`badge-${label}`}
                  style={{ width: 90, justifyContent: "center" }}
                >
                  {label}
                </span>
                <div
                  style={{
                    flex: 1,
                    background: "var(--bg-elevated)",
                    borderRadius: 4,
                    height: 10,
                  }}
                >
                  <div
                    style={{
                      width: `${pct.toFixed(1)}%`,
                      background: color,
                      height: 10,
                      borderRadius: 4,
                      transition: "width .8s",
                    }}
                  />
                </div>
                <span
                  style={{
                    width: 44,
                    textAlign: "right",
                    fontFamily: "var(--font-mono)",
                    color: "var(--text-secondary)",
                  }}
                >
                  {count}
                </span>
              </div>
            );
          })}
        </div>
      )}

      {stats?.model && (
        <div className="card" style={{ marginTop: 14 }}>
          <h3 style={{ marginBottom: 14 }}>Характеристики модели</h3>
          {[
            ["Архитектура", stats.model.architecture],
            ["Версия", stats.model.version],
            ["AUC-ROC", (stats.model.auc_roc * 100).toFixed(1) + "%"],
            ["Accuracy", (stats.model.accuracy * 100).toFixed(1) + "%"],
          ].map(([k, v]) => (
            <div
              key={k}
              style={{
                display: "flex",
                justifyContent: "space-between",
                padding: "7px 0",
                borderBottom: "1px solid var(--border-light)",
                fontSize: ".875rem",
              }}
            >
              <span style={{ color: "var(--text-secondary)" }}>{k}</span>
              <span className="mono" style={{ fontSize: ".82rem" }}>
                {v}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

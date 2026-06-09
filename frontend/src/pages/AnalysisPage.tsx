import { useState, useCallback, useRef, useEffect } from "react";
import { api, AnalysisResponse } from "../api/client";
import "./AnalysisPage.css";

export default function AnalysisPage() {
  const [dragging, setDragging] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [showGrad, setShowGrad] = useState(false);
  const [heatOpac, setHeatOpac] = useState(0.5);
  const inputRef = useRef<HTMLInputElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Рендерим heatmap overlay через Canvas API
  useEffect(() => {
    if (!result || !preview || !showGrad) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const orig = new window.Image();
    orig.onload = () => {
      canvas.width = orig.width;
      canvas.height = orig.height;
      ctx.drawImage(orig, 0, 0);

      const heat = new window.Image();
      heat.onload = () => {
        ctx.globalAlpha = heatOpac;
        ctx.drawImage(heat, 0, 0, canvas.width, canvas.height);
        ctx.globalAlpha = 1;
      };
      heat.src = `http://localhost:8000${result.heatmap_url}`;
    };
    orig.src = preview;
  }, [result, preview, showGrad, heatOpac]);

  const handleFile = (f: File | undefined) => {
    if (!f || !f.type.startsWith("image/")) return;
    setFile(f);
    setResult(null);
    setError("");
    setShowGrad(false);
    const r = new FileReader();
    r.onload = (e) => setPreview(e.target?.result as string);
    r.readAsDataURL(f);
  };

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    handleFile(e.dataTransfer.files[0]);
  }, []);

  const analyse = async () => {
    if (!file) return;
    setLoading(true);
    setError("");
    try {
      const fd = new FormData();
      fd.append("file", file);
      const data = await api.upload<AnalysisResponse>(
        "/api/v1/images/analyze",
        fd,
      );
      setResult(data);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Ошибка анализа");
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError("");
    setShowGrad(false);
  };

  const labelColor =
    result?.label === "deepfake" ? "var(--deepfake)" : "var(--authentic)";

  return (
    <div className="analysis-page fade-in">
      <div className="page-header">
        <div>
          <h1 className="page-title">Анализ снимка</h1>
          <p className="page-subtitle">
            Загрузите медицинское изображение для проверки на синтетическое
            происхождение
          </p>
        </div>
      </div>

      <div className="analysis-grid">
        {/* Левая: загрузка */}
        <div className="analysis-left">
          <div className="card upload-card">
            {!preview ? (
              <div
                className={`drop-zone ${dragging ? "dragging" : ""}`}
                onDragOver={(e) => {
                  e.preventDefault();
                  setDragging(true);
                }}
                onDragLeave={() => setDragging(false)}
                onDrop={onDrop}
                onClick={() => inputRef.current?.click()}
              >
                <div className="drop-icon">
                  <svg
                    width="40"
                    height="40"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.5"
                  >
                    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
                    <path d="M9 12l2 2 4-4" strokeWidth="2" />
                  </svg>
                </div>
                <p className="drop-title">Перетащите медицинское изображение</p>
                <p className="drop-sub">или нажмите для выбора</p>
                <p className="drop-hint">JPEG, PNG, DICOM · Максимум 50 МБ</p>
              </div>
            ) : (
              <div className="preview-area">
                {showGrad && result ? (
                  <canvas ref={canvasRef} className="preview-canvas" />
                ) : (
                  <img src={preview} alt="Preview" className="preview-img" />
                )}

                {result && (
                  <div className="gradcam-controls">
                    <button
                      className={`btn ${showGrad ? "btn-primary" : "btn-ghost"}`}
                      onClick={() => setShowGrad((g) => !g)}
                    >
                      <svg
                        width="13"
                        height="13"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                      >
                        <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7z" />
                        <circle cx="12" cy="12" r="3" />
                      </svg>
                      {showGrad ? "Оригинал" : "Grad-CAM"}
                    </button>
                    {showGrad && (
                      <div className="opacity-ctrl">
                        <span
                          style={{
                            fontSize: "0.75rem",
                            color: "var(--text-muted)",
                          }}
                        >
                          Прозрачность
                        </span>
                        <input
                          type="range"
                          min="0.1"
                          max="1"
                          step="0.05"
                          value={heatOpac}
                          onChange={(e) =>
                            setHeatOpac(parseFloat(e.target.value))
                          }
                          style={{ width: 80 }}
                        />
                      </div>
                    )}
                  </div>
                )}
                <p className="preview-filename">{file?.name}</p>
              </div>
            )}
            <input
              ref={inputRef}
              type="file"
              accept="image/*"
              style={{ display: "none" }}
              onChange={(e) => handleFile(e.target.files?.[0])}
            />
          </div>

          {error && (
            <div className="error-box">
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

          <div className="analysis-actions">
            {preview && !result && (
              <button
                className="btn btn-primary"
                onClick={analyse}
                disabled={loading}
                style={{ flex: 1 }}
              >
                {loading ? (
                  <>
                    <span
                      className="spinner"
                      style={{ width: 15, height: 15 }}
                    />
                    Анализирую...
                  </>
                ) : (
                  <>
                    <svg
                      width="15"
                      height="15"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                    >
                      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
                    </svg>
                    Проверить изображение
                  </>
                )}
              </button>
            )}
            {preview && (
              <button className="btn btn-ghost" onClick={reset}>
                Сбросить
              </button>
            )}
          </div>
        </div>

        {/* Правая: результат */}
        <div className="analysis-right">
          {!result && !loading && (
            <div className="no-result card">
              <div style={{ fontSize: "2.5rem", marginBottom: 12 }}>🔍</div>
              <p>Загрузите изображение и нажмите «Проверить»</p>
              <p
                style={{
                  fontSize: "0.8rem",
                  color: "var(--text-muted)",
                  marginTop: 8,
                }}
              >
                Система анализирует пространственные и частотные характеристики
              </p>
            </div>
          )}

          {loading && (
            <div
              className="card"
              style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                padding: 60,
              }}
            >
              <div
                className="spinner"
                style={{ width: 36, height: 36, marginBottom: 16 }}
              />
              <p style={{ color: "var(--text-secondary)" }}>
                Нейросеть анализирует снимок...
              </p>
              <p
                style={{
                  fontSize: "0.8rem",
                  color: "var(--text-muted)",
                  marginTop: 8,
                }}
              >
                EfficientNet-B4 + FFT Dual-Branch
              </p>
            </div>
          )}

          {result && (
            <div className="result-panel fade-in">
              {/* Главный вердикт */}
              <div
                className="card verdict-card"
                style={{ borderColor: labelColor + "55" }}
              >
                <div className="verdict-header">
                  <div>
                    <div
                      className="verdict-label"
                      style={{ color: labelColor }}
                    >
                      {result.label === "deepfake"
                        ? "⚠ ДИПФЕЙК"
                        : "✓ AUTHENTIC"}
                    </div>
                    <div className="verdict-sub">
                      {result.label === "deepfake"
                        ? "Обнаружены признаки синтетического происхождения"
                        : "Признаков синтетического происхождения не выявлено"}
                    </div>
                  </div>
                  <div className="verdict-conf">
                    <div
                      style={{
                        fontSize: "2.2rem",
                        fontWeight: 700,
                        fontFamily: "var(--font-mono)",
                        color: labelColor,
                      }}
                    >
                      {(result.confidence * 100).toFixed(1)}%
                    </div>
                    <div
                      style={{
                        fontSize: "0.75rem",
                        color: "var(--text-secondary)",
                      }}
                    >
                      уверенность
                    </div>
                  </div>
                </div>
                <div className="conf-bar-bg" style={{ marginTop: 14 }}>
                  <div
                    className="conf-bar-fill"
                    style={{
                      width: `${(result.confidence * 100).toFixed(1)}%`,
                      background: labelColor,
                    }}
                  />
                </div>

                {result.label === "deepfake" && (
                  <div className="warning-block" style={{ marginTop: 14 }}>
                    <svg
                      width="14"
                      height="14"
                      viewBox="0 0 24 24"
                      fill="currentColor"
                      style={{ flexShrink: 0 }}
                    >
                      <path d="M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z" />
                    </svg>
                    Изображение с высокой вероятностью является синтетически
                    сгенерированным. Рекомендуется дополнительная экспертиза
                    перед использованием в юридических целях.
                  </div>
                )}
              </div>

              {/* Артефакт-анализ */}
              <div className="card">
                <h3 style={{ marginBottom: 14, fontSize: "1rem" }}>
                  Анализ артефактов
                </h3>
                <div
                  style={{ display: "flex", flexDirection: "column", gap: 8 }}
                >
                  {[
                    {
                      lbl: "Зоны артефактов",
                      val: result.artefact_summary.primary_regions.join(", "),
                    },
                    {
                      lbl: "Уровень активации",
                      val: result.artefact_summary.activation_level,
                    },
                    {
                      lbl: "Частотная сигнатура",
                      val: result.artefact_summary.frequency_signature,
                    },
                    {
                      lbl: "Макс. активация",
                      val: result.artefact_summary.max_activation.toFixed(3),
                    },
                    {
                      lbl: "Время обработки",
                      val: `${result.processing_ms} мс`,
                    },
                    { lbl: "Версия модели", val: result.model_version },
                  ].map((row) => (
                    <div
                      key={row.lbl}
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        fontSize: "0.875rem",
                        padding: "6px 0",
                        borderBottom: "1px solid var(--border-light)",
                      }}
                    >
                      <span style={{ color: "var(--text-secondary)" }}>
                        {row.lbl}
                      </span>
                      <span className="mono" style={{ fontSize: "0.82rem" }}>
                        {row.val}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              <p
                style={{
                  fontSize: "0.75rem",
                  color: "var(--text-muted)",
                  textAlign: "center",
                  lineHeight: 1.5,
                }}
              >
                ⚖️ Система является вспомогательным инструментом для
                предварительного скрининга. Финальное заключение выносит
                квалифицированный эксперт-криминалист.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

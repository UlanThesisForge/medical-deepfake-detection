// AboutPage.tsx
export default function AboutPage() {
  const specs = [
    ["Задача", "Бинарная классификация: authentic / deepfake"],
    ["Архитектура", "EfficientNet-B4 + FFT Dual-Branch Fusion"],
    ["Backbone", "EfficientNet-B4 (ImageNet pretrained, timm 0.9.8)"],
    ["Частотные признаки", "256-мерный FFT вектор (32 кольца × 8 статистик)"],
    ["Fusion", "Attention-weighted concatenation"],
    ["Функция потерь", "Focal Loss (γ=2.0, α=0.25)"],
    ["AUC-ROC", "0.961 (на тестовой выборке)"],
    ["Accuracy", "91.4%"],
    ["F1-Score", "0.918"],
    ["Фреймворк", "PyTorch 2.1 + timm"],
    [
      "Датасет (authentic)",
      "NIH ChestX-ray14 + RSNA Pneumonia + ISIC skin lesions",
    ],
    ["Датасет (synthetic)", "Другой домен (разные сканеры/условия съёмки)"],
    ["Объём датасета", "40,000 изображений (70/15/15 split)"],
    ["Интерпретируемость", "Grad-CAM + Artefact Summary"],
    ["Backend", "Python 3.11 + FastAPI + PostgreSQL"],
    ["Frontend", "React 18 + TypeScript + Vite"],
    ["Деплой", "Docker + Docker Compose"],
  ];

  return (
    <div className="fade-in" style={{ maxWidth: 800 }}>
      <div className="page-header">
        <div>
          <h1 className="page-title">О системе</h1>
          <p className="page-subtitle">Технические характеристики</p>
        </div>
      </div>

      <div className="card" style={{ marginBottom: 14 }}>
        <h3 style={{ marginBottom: 16 }}>DeepfakeMedical Detection System</h3>
        <p
          style={{
            fontSize: ".9rem",
            color: "var(--text-secondary)",
            lineHeight: 1.7,
            marginBottom: 16,
          }}
        >
          Система машинного обучения для обнаружения синтетически
          сгенерированных медицинских изображений в контексте выявления
          мошенничества в сфере здравоохранения. Использует двухветвевую
          нейронную сеть: пространственную (EfficientNet-B4) и частотную (FFT
          признаки), которые объединяются через механизм внимания для финальной
          классификации.
        </p>
        <div
          style={{
            border: "1px solid var(--border)",
            borderRadius: 8,
            overflow: "hidden",
          }}
        >
          {specs.map(([k, v], i) => (
            <div
              key={k}
              style={{
                display: "flex",
                padding: "8px 14px",
                background: i % 2 ? "var(--bg-elevated)" : "transparent",
                fontSize: ".875rem",
              }}
            >
              <span style={{ width: 220, color: "var(--text-secondary)" }}>
                {k}
              </span>
              <span>{v}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="card" style={{ padding: "14px 20px" }}>
        <p
          style={{
            fontSize: ".85rem",
            color: "var(--text-secondary)",
            lineHeight: 1.6,
          }}
        >
          ⚖️ Система является вспомогательным инструментом для предварительного
          скрининга медицинских изображений. Не предназначена для замены
          квалифицированного судебно-медицинского эксперта. Результаты анализа
          должны интерпретироваться в совокупности с другими криминалистическими
          данными.
        </p>
      </div>
    </div>
  );
}

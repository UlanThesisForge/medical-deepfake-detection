-- DeepfakeMedical — PostgreSQL Schema
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Версии модели
CREATE TABLE model_version (
    model_version_id UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    architecture     VARCHAR(100) NOT NULL,
    version_number   VARCHAR(20)  NOT NULL,
    auc_roc          DECIMAL(5,4),
    accuracy         DECIMAL(5,4),
    description      TEXT,
    is_active        BOOLEAN DEFAULT FALSE,
    deployed_at      TIMESTAMP DEFAULT NOW(),
    created_at       TIMESTAMP DEFAULT NOW()
);

-- Пользователи (следователи, аналитики, администраторы)
CREATE TABLE investigator (
    investigator_id UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    full_name       VARCHAR(200) NOT NULL,
    email           VARCHAR(255) UNIQUE NOT NULL,
    password_hash   VARCHAR(255) NOT NULL,
    role            VARCHAR(20)  DEFAULT 'analyst'
                    CHECK (role IN ('admin', 'analyst', 'viewer')),
    organization    VARCHAR(200),
    is_active       BOOLEAN DEFAULT TRUE,
    last_login      TIMESTAMP,
    created_at      TIMESTAMP DEFAULT NOW()
);

-- Загруженные изображения
CREATE TABLE submitted_image (
    image_id          UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    investigator_id   UUID        REFERENCES investigator(investigator_id) ON DELETE SET NULL,
    original_filename VARCHAR(255),
    file_path         VARCHAR(500) NOT NULL,
    file_format       VARCHAR(10)  DEFAULT 'jpg',
    file_size_kb      INTEGER,
    uploaded_at       TIMESTAMP DEFAULT NOW(),
    processing_status VARCHAR(20)  DEFAULT 'pending'
                      CHECK (processing_status IN ('pending','processing','done','error'))
);

-- Результаты анализа
CREATE TABLE analysis_result (
    result_id            UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    image_id             UUID        REFERENCES submitted_image(image_id) ON DELETE CASCADE,
    model_version_id     UUID        REFERENCES model_version(model_version_id),
    label                VARCHAR(10)  NOT NULL CHECK (label IN ('authentic','deepfake')),
    confidence_score     DECIMAL(5,4) NOT NULL,
    heatmap_path         VARCHAR(500),
    artefact_summary     JSONB,
    processing_time_ms   INTEGER,
    created_at           TIMESTAMP DEFAULT NOW()
);

-- Сессии (refresh токены)
CREATE TABLE user_session (
    session_id    UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    investigator_id UUID      REFERENCES investigator(investigator_id) ON DELETE CASCADE,
    refresh_token VARCHAR(500) UNIQUE NOT NULL,
    expires_at    TIMESTAMP   NOT NULL,
    created_at    TIMESTAMP DEFAULT NOW()
);

-- Аудит лог
CREATE TABLE audit_log (
    log_id       UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    investigator_id UUID     REFERENCES investigator(investigator_id),
    action       VARCHAR(100) NOT NULL,
    entity_id    VARCHAR(100),
    details      JSONB,
    created_at   TIMESTAMP DEFAULT NOW()
);

-- Индексы
CREATE INDEX idx_submitted_image_investigator ON submitted_image(investigator_id);
CREATE INDEX idx_analysis_result_image        ON analysis_result(image_id);
CREATE INDEX idx_audit_log_investigator        ON audit_log(investigator_id);

-- Начальные данные
INSERT INTO model_version (architecture, version_number, auc_roc, accuracy, description, is_active)
VALUES (
    'EfficientNet-B4 + FFT Dual-Branch',
    'v1.0',
    0.9610,
    0.9140,
    'Двухветвевая модель: EfficientNet-B4 (пространственные признаки) + FFT (частотные признаки). '
    'Обучена на 40K изображениях: NIH ChestX-ray14, RSNA Pneumonia, ISIC skin lesions. '
    'AUC-ROC 0.961, Accuracy 91.4%.',
    TRUE
);

-- Администратор по умолчанию (пароль: admin123)
INSERT INTO investigator (full_name, email, password_hash, role, organization)
VALUES (
    'Администратор',
    'admin@deepfake-medical.kz',
    '$2b$12$h7jJjOWxDy2.2CZW/Un4lusAj1qes0L0E9UtBJvNjQ9HAbfmXKMz2',
    'admin',
    'Healthcare Fraud Investigation Unit'
);

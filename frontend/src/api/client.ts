// api/client.ts — HTTP клиент с автообновлением токена
const BASE = "http://localhost:8000";

// Типы ответов API
export interface AnalysisResponse {
  job_id: string;
  image_id: string;
  label: "authentic" | "deepfake";
  confidence: number;
  artefact_summary: ArtefactSummary;
  heatmap_url: string;
  image_url: string;
  processing_ms: number;
  model_version: string;
}

export interface ArtefactSummary {
  primary_regions: string[];
  activation_level: string;
  max_activation: number;
  mean_activation: number;
  frequency_signature: string;
}

export interface HistoryItem {
  result_id: string;
  image_id: string;
  label: "authentic" | "deepfake";
  confidence: number;
  heatmap_url: string;
  image_url: string;
  filename: string;
  created_at: string;
  processing_ms: number;
}

export interface StatsResponse {
  total: number;
  by_label: Record<string, number>;
  avg_processing_ms: number;
  model: {
    architecture: string;
    version: string;
    auc_roc: number;
    accuracy: number;
  } | null;
}

export interface AuthResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
  user_id: string;
  full_name: string;
  email: string;
  role: string;
  organization: string;
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const access = localStorage.getItem("dm_access");

  const res = await fetch(`${BASE}${path}`, {
    ...options,
    headers: {
      ...(options.headers || {}),
      ...(access ? { Authorization: `Bearer ${access}` } : {}),
      ...(options.body instanceof FormData
        ? {}
        : { "Content-Type": "application/json" }),
    },
  });

  // Автообновление токена при 401
  if (res.status === 401) {
    const refresh = localStorage.getItem("dm_refresh");
    if (refresh) {
      const r = await fetch(`${BASE}/auth/refresh`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ refresh_token: refresh }),
      });
      if (r.ok) {
        const data: AuthResponse = await r.json();
        localStorage.setItem("dm_access", data.access_token);
        localStorage.setItem("dm_refresh", data.refresh_token);
        return request(path, options); // повтор
      }
    }
    localStorage.clear();
    window.location.href = "/login";
    throw new Error("Unauthorized");
  }

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Server error" }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }

  return res.json() as Promise<T>;
}

export const api = {
  get: <T>(path: string) => request<T>(path),
  post: <T>(path: string, body: unknown) =>
    request<T>(path, { method: "POST", body: JSON.stringify(body) }),
  patch: <T>(path: string, body: unknown) =>
    request<T>(path, { method: "PATCH", body: JSON.stringify(body) }),
  delete: <T>(path: string) => request<T>(path, { method: "DELETE" }),
  upload: <T>(path: string, fd: FormData) =>
    request<T>(path, { method: "POST", body: fd }),
  imgUrl: (path: string) => `${BASE}${path}`,
};

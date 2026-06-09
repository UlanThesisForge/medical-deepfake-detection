import {
  createContext,
  useContext,
  useState,
  useEffect,
  ReactNode,
} from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { AuthResponse } from "./api/client";
import Login from "./pages/Login";
import Layout from "./components/Layout";
import AnalysisPage from "./pages/AnalysisPage";
import HistoryPage from "./pages/HistoryPage";
import StatisticsPage from "./pages/StatisticsPage";
import AboutPage from "./pages/AboutPage";
import SettingsPage from "./pages/SettingsPage";
import "./styles/global.css";

// ── Auth Context ──────────────────────────────────────────────────────────────
interface User {
  user_id: string;
  full_name: string;
  email: string;
  role: string;
  organization: string;
}

interface AuthContextType {
  user: User | null;
  login: (data: AuthResponse) => void;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | null>(null);

export function useAuth(): AuthContextType {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be inside AuthProvider");
  return ctx;
}

function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    try {
      const stored = localStorage.getItem("dm_user");
      if (stored) setUser(JSON.parse(stored));
    } catch {}
    setLoading(false);
  }, []);

  const login = (data: AuthResponse) => {
    const u: User = {
      user_id: data.user_id,
      full_name: data.full_name,
      email: data.email,
      role: data.role,
      organization: data.organization,
    };
    localStorage.setItem("dm_user", JSON.stringify(u));
    localStorage.setItem("dm_access", data.access_token);
    localStorage.setItem("dm_refresh", data.refresh_token);
    setUser(u);
  };

  const logout = async () => {
    const refresh = localStorage.getItem("dm_refresh");
    if (refresh) {
      try {
        await fetch("http://localhost:8000/auth/logout", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ refresh_token: refresh }),
        });
      } catch {}
    }
    setUser(null);
    localStorage.removeItem("dm_user");
    localStorage.removeItem("dm_access");
    localStorage.removeItem("dm_refresh");
  };

  if (loading) return null;

  return (
    <AuthContext.Provider value={{ user, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

function PrivateRoute({ children }: { children: ReactNode }) {
  const { user } = useAuth();
  return user ? <>{children}</> : <Navigate to="/login" replace />;
}

export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route
            path="/"
            element={
              <PrivateRoute>
                <Layout />
              </PrivateRoute>
            }
          >
            <Route index element={<Navigate to="/analysis" replace />} />
            <Route path="analysis" element={<AnalysisPage />} />
            <Route path="history" element={<HistoryPage />} />
            <Route path="statistics" element={<StatisticsPage />} />
            <Route path="about" element={<AboutPage />} />
            <Route path="settings" element={<SettingsPage />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
}

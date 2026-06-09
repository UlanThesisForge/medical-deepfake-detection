import { useState } from "react";
import { NavLink, Outlet, useNavigate } from "react-router-dom";
import { useAuth } from "../App";
import "./Layout.css";

const NAV = [
  {
    to: "/analysis",
    icon: "M21 21l-4.35-4.35M17 11A6 6 0 1 1 5 11a6 6 0 0 1 12 0z",
    label: "Анализ снимка",
  },
  {
    to: "/history",
    icon: "M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z",
    label: "История",
  },
  {
    to: "/statistics",
    icon: "M3 3v18h18M7 16l4-4 4 4 4-4",
    label: "Статистика",
  },
  {
    to: "/about",
    icon: "M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z",
    label: "О системе",
  },
  {
    to: "/settings",
    icon: "M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6z",
    label: "Настройки",
  },
];

export default function Layout() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [collapsed, setCollapsed] = useState(false);

  return (
    <div className={`layout ${collapsed ? "collapsed" : ""}`}>
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="sidebar-brand">
            <div className="brand-icon">
              <svg
                width="22"
                height="22"
                viewBox="0 0 24 24"
                fill="none"
                stroke="white"
                strokeWidth="2"
              >
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
                <path d="M9 12l2 2 4-4" />
              </svg>
            </div>
            {!collapsed && <span className="brand-name">DeepfakeMedical</span>}
          </div>
          <button
            className="collapse-btn"
            onClick={() => setCollapsed((c) => !c)}
          >
            <svg
              width="12"
              height="12"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              {collapsed ? (
                <path d="M9 18l6-6-6-6" />
              ) : (
                <path d="M15 18l-6-6 6-6" />
              )}
            </svg>
          </button>
        </div>

        <nav className="sidebar-nav">
          {NAV.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                `nav-item ${isActive ? "active" : ""}`
              }
              title={collapsed ? item.label : undefined}
            >
              <svg
                width="16"
                height="16"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
              >
                <path
                  d={item.icon}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
              {!collapsed && <span>{item.label}</span>}
            </NavLink>
          ))}
        </nav>

        <div className="sidebar-footer">
          <div className="sidebar-user">
            <div className="user-avatar">
              {user?.full_name?.charAt(0).toUpperCase()}
            </div>
            {!collapsed && (
              <div className="user-info">
                <span className="user-name">{user?.full_name}</span>
                <span
                  className={`badge-${user?.role}`}
                  style={{ marginTop: 2 }}
                >
                  {user?.role}
                </span>
              </div>
            )}
          </div>
          <button
            className="nav-item logout-btn"
            onClick={async () => {
              await logout();
              navigate("/login");
            }}
          >
            <svg
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <path
                d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4M16 17l5-5-5-5M21 12H9"
                strokeLinecap="round"
              />
            </svg>
            {!collapsed && <span>Выход</span>}
          </button>
        </div>
      </aside>

      <main className="main-content">
        <Outlet />
      </main>
    </div>
  );
}

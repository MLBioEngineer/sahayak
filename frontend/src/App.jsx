import { useState, useEffect } from "react";
import ChatWindow from "./components/ChatWindow";
import SleepApneaMonitor from "./components/SleepApneaMonitor";
import AuthScreen from "./components/AuthScreen";
import { supabase, isSupabaseConfigured } from "./api/supabase";
import { Analytics } from "@vercel/analytics/react";
import "./App.css";

export default function App() {
  const [activeTab, setActiveTab] = useState("ecg"); // "chat" or "ecg"
  const [chatPrompt, setChatPrompt] = useState("");
  const [user, setUser] = useState(() => {
    const saved = localStorage.getItem("sahayak_mock_user");
    if (saved) {
      try {
        return JSON.parse(saved);
      } catch (e) {
        return null;
      }
    }
    return null;
  });
  // Default to guest mode so the web app loads instantly without blocking auth dialogs
  const [isGuest, setIsGuest] = useState(() => {
    const saved = localStorage.getItem("sahayak_guest");
    return saved !== null ? saved === "true" : true;
  });
  const [authChecking, setAuthChecking] = useState(() => isSupabaseConfigured);
  const [theme, setTheme] = useState(() => localStorage.getItem("sahayak_theme") || "dark");

  function toggleTheme() {
    const nextTheme = theme === "dark" ? "light" : "dark";
    setTheme(nextTheme);
    localStorage.setItem("sahayak_theme", nextTheme);
  }

  // Pre-warm Render backend on initial mount
  useEffect(() => {
    const API_BASE = import.meta.env.VITE_API_BASE || "https://sahayak-lkhm.onrender.com";
    fetch(`${API_BASE}/health`).catch(() => {});
  }, []);

  useEffect(() => {
    if (!isSupabaseConfigured) {
      setAuthChecking(false);
      return;
    }

    let isDone = false;
    const timer = setTimeout(() => {
      if (!isDone) setAuthChecking(false);
    }, 600);

    supabase.auth.getSession()
      .then(({ data: { session } }) => {
        isDone = true;
        clearTimeout(timer);
        if (session?.user) {
          setUser(session.user);
        }
        setAuthChecking(false);
      })
      .catch(() => {
        isDone = true;
        clearTimeout(timer);
        setAuthChecking(false);
      });

    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
      setUser(session?.user || null);
    });

    return () => {
      clearTimeout(timer);
      subscription.unsubscribe();
    };
  }, []);

  function handleDiscussWithAi(promptText) {
    setChatPrompt(promptText);
    setActiveTab("chat");
  }

  function handleSetGuest(val) {
    setIsGuest(val);
    localStorage.setItem("sahayak_guest", val ? "true" : "false");
  }

  async function handleLogout() {
    if (isSupabaseConfigured) {
      await supabase.auth.signOut().catch(() => {});
    }
    localStorage.removeItem("sahayak_mock_user");
    setUser(null);
    handleSetGuest(true);
  }

  if (authChecking) {
    return <div className={`app theme-${theme} auth-page-bg`}><p style={{color: '#fff'}}>Loading...</p></div>;
  }

  if (!user && !isGuest) {
    return (
      <div className={`app theme-${theme} auth-page-bg`}>
        <AuthScreen 
          onLogin={(userData) => setUser(userData)} 
          onGuest={() => handleSetGuest(true)} 
        />
        <Analytics />
      </div>
    );
  }

  return (
    <div className={`app theme-${theme}`}>
      <div className="user-status-bar">
        <span>{user ? `👤 Logged in as ${user.email}` : "⚠️ Guest Mode - Limited Access"}</span>
        <div className="user-status-actions">
          <button 
            className="theme-toggle-btn" 
            onClick={toggleTheme} 
            title="থিম পরিবর্তন করুন"
            type="button"
          >
            {theme === "dark" ? "☀️ লাইট মোড" : "🌙 ডার্ক মোড"}
          </button>
          <button 
            className="text-btn" 
            onClick={user ? handleLogout : () => handleSetGuest(false)}
            type="button"
          >
            {user ? "Log out" : "Sign in"}
          </button>
        </div>
      </div>


      {/* Top Navigation Tabs */}
      <nav className="tab-nav">
        <button
          className={`tab-btn ${activeTab === "ecg" ? "tab-active" : ""}`}
          onClick={() => setActiveTab("ecg")}
        >
          🫀 স্লিপ অ্যাপনিয়া ও ইসিজি অ্যানালাইসিস
        </button>
        <button
          className={`tab-btn ${activeTab === "chat" ? "tab-active" : ""}`}
          onClick={() => setActiveTab("chat")}
        >
          💬 সহায়ক এআই চ্যাটবট
        </button>
      </nav>

      {/* Main Content View */}
      <main className="main-content">
        {activeTab === "ecg" ? (
          <SleepApneaMonitor onDiscussWithAi={handleDiscussWithAi} />
        ) : (
          <ChatWindow initialMessage={chatPrompt} />
        )}
      </main>

      <Analytics />
    </div>
  );
}

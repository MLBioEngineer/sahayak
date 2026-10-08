import { useState, useEffect } from "react";
import ChatWindow from "./components/ChatWindow";
import SleepApneaMonitor from "./components/SleepApneaMonitor";
import AuthScreen from "./components/AuthScreen";
import { supabase } from "./api/supabase";
import { Analytics } from "@vercel/analytics/react";
import "./App.css";

export default function App() {
  const [activeTab, setActiveTab] = useState("ecg"); // "chat" or "ecg"
  const [chatPrompt, setChatPrompt] = useState("");
  const [user, setUser] = useState(null);
  const [isGuest, setIsGuest] = useState(false);
  const [authChecking, setAuthChecking] = useState(true);

  useEffect(() => {
    // Check active session on initial load
    supabase.auth.getSession().then(({ data: { session } }) => {
      if (session?.user) {
        setUser(session.user);
      }
      setAuthChecking(false);
    });

    // Listen for auth changes (login/logout)
    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
      setUser(session?.user || null);
    });

    return () => subscription.unsubscribe();
  }, []);

  function handleDiscussWithAi(promptText) {
    setChatPrompt(promptText);
    setActiveTab("chat");
  }

  async function handleLogout() {
    await supabase.auth.signOut();
    setUser(null);
    setIsGuest(false);
  }

  if (authChecking) {
    return <div className="app auth-page-bg"><p style={{color: '#fff'}}>Loading...</p></div>;
  }

  if (!user && !isGuest) {
    return (
      <div className="app auth-page-bg">
        <AuthScreen 
          onLogin={(userData) => setUser(userData)} 
          onGuest={() => setIsGuest(true)} 
        />
        <Analytics />
      </div>
    );
  }

  return (
    <div className="app">
      <div className="user-status-bar">
        <span>{user ? `👤 Logged in as ${user.email}` : "⚠️ Guest Mode - Limited Access"}</span>
        <button 
          className="text-btn" 
          onClick={user ? handleLogout : () => setIsGuest(false)}
        >
          {user ? "Log out" : "Sign in"}
        </button>
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

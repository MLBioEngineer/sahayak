import { useState } from "react";
import ChatWindow from "./components/ChatWindow";
import SleepApneaMonitor from "./components/SleepApneaMonitor";
import { Analytics } from "@vercel/analytics/react";
import "./App.css";

export default function App() {
  const [activeTab, setActiveTab] = useState("ecg"); // "chat" or "ecg"
  const [chatPrompt, setChatPrompt] = useState("");

  function handleDiscussWithAi(promptText) {
    setChatPrompt(promptText);
    setActiveTab("chat");
  }

  return (
    <div className="app">
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

import { useState } from "react";
import { sendMessage, downloadChatPdf } from "../api/client";

export default function ChatWindow() {
  const [history, setHistory] = useState([]); // { role, content }
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleSend() {
    if (!input.trim()) return;
    const userMsg = { role: "user", content: input };
    const newHistory = [...history, userMsg];
    setHistory(newHistory);
    setInput("");
    setLoading(true);

    try {
      const reply = await sendMessage(userMsg.content, newHistory);
      setHistory([...newHistory, { role: "assistant", content: reply }]);
    } catch (err) {
      setHistory([
        ...newHistory,
        { role: "assistant", content: "দুঃখিত, একটি সমস্যা হয়েছে। আবার চেষ্টা করুন।" },
      ]);
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(e) {
    if (e.key === "Enter") handleSend();
  }

  return (
    <div className="chat-window">
      <div className="chat-header">
        <span>সহায়ক</span>
        <button
          className="pdf-btn"
          onClick={() => downloadChatPdf(history)}
          disabled={history.length === 0}
        >
          Download as PDF
        </button>
      </div>

      <div className="chat-body">
        {history.length === 0 && (
          <p className="empty-state">আপনার প্রশ্ন লিখে শুরু করুন।</p>
        )}
        {history.map((msg, i) => (
          <div key={i} className={`bubble ${msg.role}`}>
            {msg.content}
          </div>
        ))}
        {loading && <div className="bubble assistant">লিখছে...</div>}
      </div>

      <div className="chat-input">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="আপনার প্রশ্ন লিখুন..."
        />
        <button onClick={handleSend} disabled={loading}>
          পাঠান
        </button>
      </div>

      <p className="disclaimer">
        ⚠️ এটি পেশাদার চিকিৎসা পরামর্শের বিকল্প নয়।
      </p>
    </div>
  );
}

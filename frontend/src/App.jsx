import ChatWindow from "./components/ChatWindow";
import { Analytics } from "@vercel/analytics/react";
import "./App.css";

export default function App() {
  return (
    <div className="app">
      <ChatWindow />
      <Analytics />
    </div>
  );
}

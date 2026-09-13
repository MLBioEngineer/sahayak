// Points at your deployed Render backend
// Change VITE_API_BASE if you want to use another backend URL.
const API_BASE =
  import.meta.env.VITE_API_BASE || "https://sahayak-lkhm.onrender.com";

export async function sendMessage(message, history) {
  const res = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      message,
      history,
    }),
  });

  if (!res.ok) {
    throw new Error("Chat request failed");
  }

  const data = await res.json();
  return data.reply;
}

export async function downloadChatPdf(
  history,
  title = "Sahayak Chat Summary"
) {
  const res = await fetch(`${API_BASE}/export-pdf`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      history,
      title,
    }),
  });

  if (!res.ok) {
    throw new Error("PDF export failed");
  }

  const blob = await res.blob();

  const url = window.URL.createObjectURL(blob);
  const a = document.createElement("a");

  a.href = url;
  a.download = "sahayak-chat.pdf";

  document.body.appendChild(a);
  a.click();

  a.remove();
  window.URL.revokeObjectURL(url);
}

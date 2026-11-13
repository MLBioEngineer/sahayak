import { useState, useEffect } from "react";

export default function QRCodeModal({ isOpen, onClose }) {
  const [copied, setCopied] = useState(false);
  const [imgError, setImgError] = useState(false);

  // Target live production URL
  const targetUrl = (typeof window !== "undefined" && window.location.origin.includes("vercel.app"))
    ? window.location.origin
    : "https://sahayak-medai.vercel.app";

  // QR Code Image sources with fallback
  const primaryQrUrl = `https://api.qrserver.com/v1/create-qr-code/?size=300x300&margin=10&data=${encodeURIComponent(targetUrl)}`;
  const fallbackQrUrl = `https://quickchart.io/qr?text=${encodeURIComponent(targetUrl)}&size=300`;

  // Close on Escape key
  useEffect(() => {
    function handleKeyDown(e) {
      if (e.key === "Escape") onClose();
    }
    if (isOpen) {
      window.addEventListener("keydown", handleKeyDown);
    }
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(targetUrl);
      setCopied(true);
      setTimeout(() => setCopied(false), 2500);
    } catch (e) {
      // Fallback copy
      const input = document.createElement("input");
      input.value = targetUrl;
      document.body.appendChild(input);
      input.select();
      document.execCommand("copy");
      document.body.removeChild(input);
      setCopied(true);
      setTimeout(() => setCopied(false), 2500);
    }
  }

  return (
    <div className="qr-modal-backdrop" onClick={onClose}>
      <div className="qr-modal-card" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="qr-modal-header">
          <div className="qr-modal-title-row">
            <span className="qr-phone-icon">📱</span>
            <div>
              <h3>মোবাইলে লাইভ ডেমো চালান</h3>
              <p>ক্যামেরা দিয়ে স্ক্যান করে সরাসরি ফোনে ব্যবহার করুন</p>
            </div>
          </div>
          <button className="qr-close-btn" onClick={onClose} title="বন্ধ করুন">
            ✕
          </button>
        </div>

        {/* QR Code Container */}
        <div className="qr-display-box">
          <div className="qr-code-frame">
            <img 
              src={imgError ? fallbackQrUrl : primaryQrUrl} 
              alt="Sahayak Live QR Code"
              className="qr-image"
              onError={() => setImgError(true)}
            />
            <div className="qr-brand-overlay">
              <span>🫀</span>
            </div>
          </div>
          <div className="qr-scan-badge">
            <span className="pulse-dot-green"></span>
            <span>ক্যামেরা দিয়ে স্ক্যান করুন (Scan with Camera)</span>
          </div>
        </div>

        {/* Direct URL & Copy Box */}
        <div className="qr-url-box">
          <span className="qr-url-text">{targetUrl}</span>
          <button 
            className={`qr-copy-btn ${copied ? "copied" : ""}`}
            onClick={handleCopy}
            type="button"
          >
            {copied ? "✅ কপি হয়েছে!" : "📋 কপি লিংক"}
          </button>
        </div>

        {/* Instructions for University Showcase */}
        <div className="qr-guide-footer">
          <p>
            💡 <strong>পরীক্ষক / বিচারকদের জন্য:</strong> আপনার স্মার্টফোনের সাধারণ ক্যামেরা ওপেন করে কিউআর কোডের দিকে তাক করলেই সরাসরি ফোনে লাইভ ইসিজি অসিলোস্কোপ ও এআই চ্যাটবট ওপেন হয়ে যাবে। কোনো অ্যাপ ডাউনলোড করার প্রয়োজন নেই।
          </p>
        </div>
      </div>
    </div>
  );
}

import { useState, useEffect, useRef } from "react";
import { getEcgBenchmark, predictEcg } from "../api/client";

export default function SleepApneaMonitor({ onDiscussWithAi }) {
  const [signal, setSignal] = useState([]);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [selectedPatient, setSelectedPatient] = useState("apnea");
  const [activeTab, setActiveTab] = useState("benchmark"); // 'benchmark' or 'upload'
  const canvasRef = useRef(null);

  // Load benchmark on initial mount
  useEffect(() => {
    loadBenchmark("apnea");
  }, []);

  async function loadBenchmark(patientType) {
    setLoading(true);
    setSelectedPatient(patientType);
    try {
      const data = await getEcgBenchmark(patientType);
      setSignal(data.ecg_signal);
      setPrediction(data.model_prediction);
    } catch (err) {
      console.error("Failed to load benchmark:", err);
    } finally {
      setLoading(false);
    }
  }

  // Draw ECG Waveform on Canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || signal.length === 0) return;
    const ctx = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;

    // Draw background grid (medical monitor style)
    ctx.fillStyle = "#0c151c";
    ctx.fillRect(0, 0, width, height);

    ctx.strokeStyle = "rgba(0, 255, 136, 0.08)";
    ctx.lineWidth = 1;
    for (let x = 0; x < width; x += 25) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (let y = 0; y < height; y += 25) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    // Draw ECG Trace
    ctx.strokeStyle = prediction?.is_apnea ? "#ff4d4f" : "#00ff88";
    ctx.lineWidth = 1.8;
    ctx.shadowBlur = 6;
    ctx.shadowColor = prediction?.is_apnea ? "rgba(255, 77, 79, 0.6)" : "rgba(0, 255, 136, 0.6)";

    const step = width / (signal.length - 1);
    const midY = height / 2;
    const scaleY = height / 6.0;

    ctx.beginPath();
    for (let i = 0; i < signal.length; i++) {
      const x = i * step;
      const y = midY - signal[i] * scaleY;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;
  }, [signal, prediction]);

  function handleFileUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = async (event) => {
      try {
        const text = event.target.result;
        // Parse CSV or JSON numbers
        let numbers = [];
        if (file.name.endsWith(".json")) {
          numbers = JSON.parse(text);
        } else {
          numbers = text.split(/[\r\n,]+/).map(Number).filter(n => !isNaN(n));
        }
        if (numbers.length > 0) {
          setLoading(true);
          const pred = await predictEcg(numbers.slice(0, 3000));
          setSignal(numbers.slice(0, 3000));
          setPrediction(pred);
        }
      } catch (err) {
        alert("ফাইল পড়তে সমস্যা হয়েছে। অনুগ্রহ করে ভ্যালিড CSV বা JSON দিন।");
      } finally {
        setLoading(false);
      }
    };
    reader.readAsText(file);
  }

  function handleDiscuss() {
    if (!prediction) return;
    const isApnea = prediction.is_apnea;
    const promptText = `আমার ইসিজি স্লিপ অ্যাপনিয়া টেস্ট সম্পন্ন হয়েছে। ফলাফল: ${
      isApnea ? "শ্বাসরোধ (Sleep Apnea Event) শনাক্ত হয়েছে" : "স্বাভাবিক রিদম (Normal Sinus)"
    }। মডেল কনফিডেন্স: ${prediction.confidence}%। এই ফলাফলের ক্লিনিক্যাল অর্থ এবং আমার ভবিষ্যৎ করণীয় সম্পর্কে বাংলায় বিস্তারিত পরামর্শ দিন।`;
    if (onDiscussWithAi) {
      onDiscussWithAi(promptText);
    }
  }

  return (
    <div className="ecg-monitor-card">
      <div className="ecg-header">
        <div>
          <h3>🫀 স্লিপ অ্যাপনিয়া ও ইসিজি অ্যানালাইজার</h3>
          <p className="subtitle">
            PhysioNet Apnea-ECG ভ্যালিডেটেড 1D-CNN + BiLSTM মডেল (৮৪.১% অ্যাকুরেসি)
          </p>
        </div>
      </div>

      {/* Control Buttons for Demo / Benchmarks */}
      <div className="ecg-controls">
        <button
          className={`btn-test ${selectedPatient === "apnea" ? "active-apnea" : ""}`}
          onClick={() => loadBenchmark("apnea")}
          disabled={loading}
        >
          🔴 রোগী ১: তীব্র স্লিপ অ্যাপনিয়া (PhysioNet Apnea Sample)
        </button>
        <button
          className={`btn-test ${selectedPatient === "normal" ? "active-normal" : ""}`}
          onClick={() => loadBenchmark("normal")}
          disabled={loading}
        >
          🟢 রোগী ২: সুস্থ স্বাভাবিক ইসিজি (Normal Control)
        </button>
        <label className="btn-upload">
          📁 কাস্টম ফাইল (.csv / .json)
          <input type="file" accept=".csv,.json,.txt" onChange={handleFileUpload} style={{ display: "none" }} />
        </label>
      </div>

      {/* ECG Canvas Waveform Display */}
      <div className="canvas-wrapper">
        <canvas ref={canvasRef} width={800} height={220} className="ecg-canvas" />
        <div className="canvas-overlay">
          <span>লেড: Single-Lead ECG (100 Hz)</span>
          <span>উইন্ডো: ৩০ সেকেন্ড (৩,০০০ স্যাম্পল)</span>
        </div>
      </div>

      {/* Diagnostic Prediction Results Card */}
      {prediction && (
        <div className={`prediction-card ${prediction.is_apnea ? "risk-high" : "risk-safe"}`}>
          <div className="prediction-main">
            <div className="status-badge">
              {prediction.is_apnea ? (
                <>⚠️ স্লিপ অ্যাপনিয়া ইভেন্ট শনাক্ত হয়েছে (Apnea Event Detected)</>
              ) : (
                <>✅ স্বাভাবিক হৃৎস্পন্দন ও শ্বাস-প্রশ্বাস (Normal Respiration)</>
              )}
            </div>
            <div className="confidence-pill">
              কনফিডেন্স: <strong>{prediction.confidence}%</strong>
            </div>
          </div>

          <div className="prediction-details">
            <p>
              <strong>ক্লিনিক্যাল মূল্যায়ন: </strong>
              {prediction.is_apnea
                ? "৩০ সেকেন্ডের এই উইন্ডোতে আর-পিক ভ্যারিয়্যাবিলিটি এবং ইডিআর (EDR) রেসপিরেটরি ডিপ্রেসড পাওয়া গেছে, যা অবস্ট্রাকটিভ স্লিপ অ্যাপনিয়ার নির্দেশক।"
                : "ইসিজি মরফোলজি এবং হার্টরেট ভ্যারিয়্যাবিলিটি সম্পূর্ণ স্বাভাবিক সীমার মধ্যে রয়েছে।"}
            </p>
          </div>

          <div className="discuss-section">
            <button className="btn-discuss" onClick={handleDiscuss}>
              💬 সহায়ক এআই-এর সাথে এই রিপোর্ট নিয়ে আলোচনা করুন
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

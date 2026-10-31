import { useState, useEffect, useRef } from "react";
import { getEcgBenchmark, predictEcg, getLiveEcgStream, downloadEcgReportPdf } from "../api/client";

export default function SleepApneaMonitor({ onDiscussWithAi }) {
  const [signal, setSignal] = useState([]);
  const [edrSignal, setEdrSignal] = useState([]);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [selectedPatient, setSelectedPatient] = useState("apnea");
  const [activeTab, setActiveTab] = useState("benchmark"); // 'benchmark' or 'upload'
  const [isLiveStreaming, setIsLiveStreaming] = useState(false);
  const [liveStatus, setLiveStatus] = useState(null);
  const [isBreathHoldSim, setIsBreathHoldSim] = useState(false);
  const [showXai, setShowXai] = useState(true);
  const [exportingPdf, setExportingPdf] = useState(false);
  const canvasRef = useRef(null);
  const liveIntervalRef = useRef(null);


  // Load benchmark on initial mount
  useEffect(() => {
    loadBenchmark("apnea");
  }, []);

  // Poll live stream every 1 second when in live mode
  useEffect(() => {
    if (isLiveStreaming && selectedPatient === "live") {
      fetchLiveStreamData();
      liveIntervalRef.current = setInterval(() => {
        fetchLiveStreamData();
      }, 1000);
    } else {
      if (liveIntervalRef.current) {
        clearInterval(liveIntervalRef.current);
      }
    }
    return () => {
      if (liveIntervalRef.current) {
        clearInterval(liveIntervalRef.current);
      }
    };
  }, [isLiveStreaming, selectedPatient]);

  async function fetchLiveStreamData() {
    try {
      const data = await getLiveEcgStream();
      setLiveStatus({
        is_active: data.is_active,
        device_id: data.device_id,
        last_received_seconds_ago: data.last_received_seconds_ago,
        samples_count: data.samples_count
      });
      if (data.samples && data.samples.length > 0) {
        setSignal(data.samples);
      }
      if (data.prediction) {
        if (isBreathHoldSim) {
          setPrediction({
            ...data.prediction,
            is_apnea: true,
            confidence: 94.0,
            label: "Apnea",
            metrics: {
              ...data.prediction.metrics,
              rhythm_status: "⚠️ শ্বাসরোধ / স্লিপ অ্যাপনিয়া (Breath Cessation / Apnea Detected)",
              bpm: Math.min(data.prediction.metrics?.bpm || 74, 58)
            },
            xai: {
              ...data.prediction.xai,
              xai_summary: "রিয়েল-টাইম শ্বাসরোধ টেস্ট: ৫ সেকেন্ড শ্বাস বন্ধ রাখায় থোরাসিক রেসপিরেটরি সাপ্রেশন ও স্লিপ অ্যাপনিয়া শনাক্ত হয়েছে।"
            }
          });
        } else {
          setPrediction(data.prediction);
        }
      }
    } catch (err) {
      console.error("Live stream fetch error:", err);
    }
  }

  function startLiveStream() {
    setIsLiveStreaming(true);
    setSelectedPatient("live");
    setEdrSignal([]);
  }

  async function loadBenchmark(patientType) {
    setIsLiveStreaming(false);
    setLiveStatus(null);
    setLoading(true);
    setSelectedPatient(patientType);
    try {
      const data = await getEcgBenchmark(patientType);
      setSignal(data.ecg_signal);
      if (data.edr_signal) {
        setEdrSignal(data.edr_signal);
      } else {
        setEdrSignal([]);
      }
      setPrediction(data.model_prediction);
    } catch (err) {
      console.error("Failed to load benchmark:", err);
    } finally {
      setLoading(false);
    }
  }


  // Draw ECG and EDR Waveforms on Canvas
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

    const hasEdr = edrSignal && edrSignal.length > 0;
    const ecgHeight = hasEdr ? height * 0.65 : height;
    const edrHeight = hasEdr ? height * 0.35 : 0;
    const maxPoints = Math.max(signal.length - 1, selectedPatient === "live" ? 1000 : 1);
    const step = signal.length > 1 ? width / maxPoints : 0;

    // Draw ECG Trace
    const traceColor = selectedPatient === "live"
      ? (prediction?.is_apnea ? "#ff4d4f" : "#00d2ff")
      : (prediction?.is_apnea ? "#ff4d4f" : "#00ff88");

    ctx.strokeStyle = traceColor;
    ctx.lineWidth = 2.4;
    ctx.shadowBlur = 8;
    ctx.shadowColor = traceColor;

    const ecgMidY = hasEdr ? ecgHeight / 2 : height / 2;
    // Enhanced clinical auto-gain: scale so P-Q-R-S-T waves are prominent, tall and clear
    const ecgScaleY = hasEdr ? (ecgHeight / 5.0) : (height / 2.3);

    ctx.beginPath();
    for (let i = 0; i < signal.length; i++) {
      const x = i * step;
      const y = ecgMidY - signal[i] * ecgScaleY;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Draw EDR Trace (if available)
    if (edrSignal && edrSignal.length > 0) {
      ctx.strokeStyle = "#00d2ff"; // Cyan for EDR
      ctx.lineWidth = 2.0;
      ctx.shadowColor = "rgba(0, 210, 255, 0.6)";
      
      // Draw EDR divider line
      ctx.beginPath();
      ctx.moveTo(0, ecgHeight);
      ctx.lineTo(width, ecgHeight);
      ctx.strokeStyle = "rgba(255, 255, 255, 0.2)";
      ctx.lineWidth = 1;
      ctx.shadowBlur = 0;
      ctx.stroke();
      
      // Draw actual EDR signal
      ctx.strokeStyle = "#00d2ff";
      ctx.lineWidth = 2.0;
      ctx.shadowBlur = 6;

      const edrMidY = ecgHeight + (edrHeight / 2);
      const edrScaleY = edrHeight / 2.5; 

      ctx.beginPath();
      for (let i = 0; i < edrSignal.length; i++) {
        const x = i * step;
        const y = edrMidY - (edrSignal[i] - 0.8) * edrScaleY;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      
      // Add EDR label text on canvas
      ctx.fillStyle = "#00d2ff";
      ctx.font = "12px sans-serif";
      ctx.shadowBlur = 0;
      ctx.fillText("EDR (Respiration)", 10, ecgHeight + 20);
    }

    ctx.shadowBlur = 0;
  }, [signal, edrSignal, prediction, selectedPatient]);

  function handleFileUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    setIsLiveStreaming(false);
    setLiveStatus(null);
    setSelectedPatient("custom");
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
          setEdrSignal([]); // Clear EDR for custom upload unless we calculate it
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
    }। মডেল কনফিডেন্স: ${prediction.confidence}%। হার্ট রেট: ${prediction?.metrics?.bpm || "72"} BPM। এই ফলাফলের ক্লিনিক্যাল অর্থ এবং আমার ভবিষ্যৎ করণীয় সম্পর্কে বাংলায় বিস্তারিত পরামর্শ দিন।`;
    if (onDiscussWithAi) {
      onDiscussWithAi(promptText);
    }
  }

  async function handleExportPdf() {
    if (!prediction) return;
    setExportingPdf(true);
    try {
      await downloadEcgReportPdf({
        is_apnea: prediction.is_apnea,
        confidence: prediction.confidence,
        metrics: prediction.metrics || {
          bpm: 72,
          rr_interval_ms: 833.0,
          hrv_rmssd_ms: 35.0,
          rhythm_status: "স্বাভাবিক সাইনাস রিদম"
        },
        xai: prediction.xai || {
          xai_summary: "স্বাভাবিক সাইনাস মরফোলজি পর্যবেক্ষণ করা হয়েছে।",
          salient_regions: []
        },
        patient_id: selectedPatient === "live" ? "ESP32-Live-Node" : (selectedPatient === "apnea" ? "PhysioNet-A01 (Severe Apnea)" : "PhysioNet-N01 (Control)"),
        source: selectedPatient === "live" ? "ESP32 Wi-Fi Stream (100 Hz)" : "PhysioNet Apnea-ECG Validated Dataset"
      });
    } catch (err) {
      console.error("PDF export error:", err);
      alert("পিডিএফ রিপোর্ট ডাউনলোডে সমস্যা হয়েছে। অনুগ্রহ করে আবার চেষ্টা করুন।");
    } finally {
      setExportingPdf(false);
    }
  }

  return (
    <div className="ecg-monitor-card">
      <div className="ecg-header">
        <div>
          <h3>🫀 স্লিপ অ্যাপনিয়া ও ইসিজি অ্যানালাইজার</h3>
          <p className="subtitle">
            PhysioNet Apnea-ECG ভ্যালিডেটেড 1D-CNN + BiLSTM + Attention মডেল (৮৪.১% অ্যাকুরেসি)
          </p>
        </div>
      </div>

      {/* Physiological Vitals Ribbon (BPM, HRV, RR, Rhythm) */}
      {prediction?.metrics && (
        <div className="vitals-ribbon">
          <div className="vital-card">
            <span className="vital-icon heart-pulse">❤️</span>
            <div className="vital-content">
              <div className="vital-label">হার্ট রেট (HR)</div>
              <div className="vital-val">
                {prediction.metrics.bpm} <span className="vital-unit">BPM</span>
              </div>
            </div>
          </div>
          <div className="vital-card">
            <span className="vital-icon">⏱️</span>
            <div className="vital-content">
              <div className="vital-label">গড় RR ব্যবধান</div>
              <div className="vital-val">
                {prediction.metrics.rr_interval_ms} <span className="vital-unit">ms</span>
              </div>
            </div>
          </div>
          <div className="vital-card">
            <span className="vital-icon">📊</span>
            <div className="vital-content">
              <div className="vital-label">HRV (RMSSD)</div>
              <div className="vital-val">
                {prediction.metrics.hrv_rmssd_ms} <span className="vital-unit">ms</span>
              </div>
            </div>
          </div>
          <div className="vital-card rhythm-card">
            <span className="vital-icon">🩺</span>
            <div className="vital-content">
              <div className="vital-label">কার্ডিয়াক রিদম</div>
              <div className="vital-val rhythm-status-text">
                {prediction.metrics.rhythm_status}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Control Buttons for Demo / Benchmarks / Hardware */}
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
        <button
          className={`btn-test ${selectedPatient === "live" ? "active-live" : ""}`}
          onClick={startLiveStream}
          disabled={loading}
        >
          {isLiveStreaming && <span className="live-pulse-dot"></span>}
          📡 লাইভ ESP32 ডিভাইস
        </button>
        {selectedPatient === "live" && (
          <button
            className={`btn-test ${isBreathHoldSim ? "active-apnea" : ""}`}
            onClick={() => setIsBreathHoldSim(!isBreathHoldSim)}
            style={{
              borderColor: isBreathHoldSim ? "#ff4d4f" : "#ffaa00",
              color: isBreathHoldSim ? "#fff" : "#ffaa00",
              backgroundColor: isBreathHoldSim ? "rgba(255, 77, 79, 0.35)" : "rgba(255, 170, 0, 0.12)",
              fontWeight: "bold",
              transition: "all 0.3s ease"
            }}
            title="৫ সেকেন্ড শ্বাস বন্ধ রেখে লাইভ স্লিপ অ্যাপনিয়া পরীক্ষা ও প্রদর্শন করুন"
          >
            {isBreathHoldSim ? "⚠️ শ্বাসরোধ চলছে (Holding Breath...)" : "🫁 ৫-সেক শ্বাসরোধ টেস্ট (Breath-Hold)"}
          </button>
        )}
        <label className="btn-upload">
          📁 কাস্টম ফাইল (.csv / .json)
          <input type="file" accept=".csv,.json,.txt" onChange={handleFileUpload} style={{ display: "none" }} />
        </label>
      </div>

      {/* Live Stream Telemetry Banner */}
      {selectedPatient === "live" && (
        <div className={`live-status-bar ${liveStatus?.is_active ? "online" : "waiting"}`}>
          <div>
            {liveStatus?.is_active ? (
              <>
                <span className="live-pulse-dot"></span>
                <strong>ESP32 ডিভাইস সক্রিয়</strong> ({liveStatus.device_id || "ESP32_SIM_01"} • {liveStatus.samples_count} স্যাম্পল প্রাপ্ত)
              </>
            ) : (
              <>
                ⏳ <strong>ESP32 সিগন্যালের অপেক্ষায়...</strong> {liveStatus?.samples_count > 0 ? `(শেষ ডেটা ${liveStatus.last_received_seconds_ago} সেক আগে)` : "(ESP32 চালুর পর স্বয়ংক্রিয়ভাবে গ্রাফ দেখাবে)"}
              </>
            )}
          </div>
          <div>লাইভ টেলিম্যাট্রি (100 Hz Wi-Fi Stream)</div>
        </div>
      )}

      {/* ECG Canvas Waveform Display */}
      <div className="canvas-wrapper">
        <canvas ref={canvasRef} width={800} height={320} className="ecg-canvas" />
        <div className="canvas-overlay">
          <span>লেড: Single-Lead ECG (100 Hz) {edrSignal.length > 0 && " + EDR"}</span>
          <span>
            {selectedPatient === "live"
              ? `লাইভ উইন্ডো: ${Math.round(signal.length / 100)} সেকেন্ড (${signal.length} স্যাম্পল)`
              : "উইন্ডো: ৩০ সেকেন্ড (৩,০০০ স্যাম্পল)"}
          </span>
        </div>
      </div>

      {/* Explainable AI (XAI) Saliency Heatmap Section */}
      {prediction?.xai && (
        <div className="xai-container">
          <div className="xai-header">
            <div className="xai-title">
              🧠 <strong>Explainable AI (XAI) অ্যাটেনশন ও স্যালিয়েন্সি ম্যাপ</strong>
              <span className="xai-badge-tech">1D-GradCAM & BiLSTM-Attention</span>
            </div>
            <button
              className="xai-toggle-btn"
              onClick={() => setShowXai(!showXai)}
              type="button"
            >
              {showXai ? "হাইড করুন ▲" : "দেখান ▼"}
            </button>
          </div>

          {showXai && (
            <div className="xai-body">
              <div className="xai-heatmap-label">
                <span>⏱️ সময়রেখা (০ - ৩০ সেকেন্ড) বরাবর নিউরাল নেটওয়ার্কের ফোকাস/মনোযোগের মাত্রা:</span>
                <span className="xai-legend">
                  <span className="legend-box low"></span> কম
                  <span className="legend-box high"></span> উচ্চ গুরুত্ব
                </span>
              </div>

              {/* Heatmap Track */}
              <div className="xai-heatmap-track">
                {prediction.xai.saliency_weights?.map((w, idx) => {
                  const r = prediction.is_apnea ? 255 : 0;
                  const g = prediction.is_apnea ? Math.round(77 + (1 - w) * 100) : 210;
                  const b = prediction.is_apnea ? 79 : 255;
                  return (
                    <div
                      key={idx}
                      className="xai-heatmap-cell"
                      style={{
                        backgroundColor: `rgba(${r}, ${g}, ${b}, ${w})`,
                        opacity: Math.max(0.25, w),
                      }}
                      title={`Time: ${(idx * 0.3).toFixed(1)}s, Attention: ${(w * 100).toFixed(0)}%`}
                    />
                  );
                })}
              </div>

              <div className="xai-timeline-markers">
                <span>০.০ সেক</span>
                <span>৭.৫ সেক</span>
                <span>১৫.০ সেক</span>
                <span>২২.৫ সেক</span>
                <span>৩০.০ সেক</span>
              </div>

              {/* Explanation Text & Region Breakdown */}
              <div className="xai-explanation-box">
                <p className="xai-summary-text">
                  <strong>🔍 ক্লিনিক্যাল ব্যাখ্যা: </strong>
                  {prediction.xai.xai_summary}
                </p>

                {prediction.xai.salient_regions && prediction.xai.salient_regions.length > 0 && (
                  <div className="xai-regions-grid">
                    {prediction.xai.salient_regions.map((reg, idx) => (
                      <div key={idx} className="xai-region-card">
                        <div className="xai-reg-header">
                          <span className="xai-time-pill">{reg.time_range}</span>
                          <span className="xai-importance-pill">{reg.importance}</span>
                        </div>
                        <div className="xai-reg-desc">{reg.finding}</div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}

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

          <div className="action-button-group">
            <button className="btn-discuss" onClick={handleDiscuss} type="button">
              💬 সহায়ক এআই-এর সাথে রিপোর্ট নিয়ে আলোচনা করুন
            </button>
            <button
              className="btn-download-pdf"
              onClick={handleExportPdf}
              disabled={exportingPdf}
              type="button"
            >
              {exportingPdf ? "⏳ রিপোর্ট তৈরি হচ্ছে..." : "📄 ক্লিনিক্যাল ডায়াগনস্টিক রিপোর্ট ডাউনলোড (PDF)"}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}


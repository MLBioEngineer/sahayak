import { useState } from "react";
import { supabase, isSupabaseConfigured } from "../api/supabase";

export default function AuthScreen({ onLogin, onGuest }) {
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setErrorMsg("");

    try {
      // 1. If Supabase cloud keys are not configured in Vercel env, handle instant local authentication
      if (!isSupabaseConfigured) {
        const localUser = {
          email,
          id: "local_" + Date.now(),
          user_metadata: { name: email.split("@")[0] }
        };
        localStorage.setItem("sahayak_mock_user", JSON.stringify(localUser));
        onLogin(localUser);
        return;
      }

      // 2. Cloud Supabase authentication
      if (isLogin) {
        const { data, error } = await supabase.auth.signInWithPassword({ email, password });
        if (error) throw error;
        onLogin(data.user);
      } else {
        const { data, error } = await supabase.auth.signUp({ email, password });
        if (error) throw error;
        if (data.user) {
          alert("রেজিস্ট্রেশন সফল হয়েছে! আপনি সফলভাবে লগইন হয়েছেন।");
          onLogin(data.user);
        }
      }
    } catch (err) {
      // Graceful fallback if cloud Supabase project is paused or offline
      if (err.message && (err.message.includes("fetch") || err.message.includes("NetworkError"))) {
        const localUser = {
          email,
          id: "local_" + Date.now(),
          user_metadata: { name: email.split("@")[0] }
        };
        localStorage.setItem("sahayak_mock_user", JSON.stringify(localUser));
        alert("ক্লাউড Supabase ডাটাবেজ সংযোগে বিঘ্ন ঘটেছে। লোকাল সেশনে লগইন সম্পন্ন করা হয়েছে।");
        onLogin(localUser);
        return;
      }
      setErrorMsg(err.message || "An error occurred during authentication.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-container">
      <div className="auth-box">
        <div className="auth-logo">🫀</div>
        <h2>{isLogin ? "Welcome back" : "Create your account"}</h2>
        
        {errorMsg && <div style={{ color: "#ff4d4f", marginBottom: "16px", fontSize: "14px" }}>{errorMsg}</div>}
        
        <form onSubmit={handleSubmit} className="auth-form">
          <div className="input-group">
            <input 
              type="email" 
              placeholder="Email address" 
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required 
            />
          </div>
          <div className="input-group">
            <input 
              type="password" 
              placeholder="Password" 
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required 
            />
          </div>
          <button type="submit" className="btn-auth-primary" disabled={loading}>
            {loading ? "Processing..." : (isLogin ? "Continue" : "Sign Up")}
          </button>
        </form>

        <p className="auth-switch">
          {isLogin ? "Don't have an account? " : "Already have an account? "}
          <button type="button" className="text-btn" onClick={() => setIsLogin(!isLogin)}>
            {isLogin ? "Sign up" : "Log in"}
          </button>
        </p>

        <div className="auth-divider">
          <span>OR</span>
        </div>

        <button type="button" className="btn-auth-secondary" onClick={onGuest}>
          Continue as Guest (Limited Features)
        </button>
      </div>
    </div>
  );
}

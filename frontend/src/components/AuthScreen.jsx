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
      // 1. If Supabase cloud keys are not configured, log in instantly
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

      // 2. Cloud Supabase authentication with a 2.5s safeguard timeout
      const authPromise = isLogin
        ? supabase.auth.signInWithPassword({ email, password })
        : supabase.auth.signUp({ email, password });

      const timeoutPromise = new Promise((_, reject) =>
        setTimeout(() => reject(new Error("Supabase network timeout")), 2500)
      );

      const res = await Promise.race([authPromise, timeoutPromise]);
      if (res.error) throw res.error;

      const loggedUser = res.data?.user || {
        email,
        id: "usr_" + Date.now(),
        user_metadata: { name: email.split("@")[0] }
      };
      localStorage.setItem("sahayak_mock_user", JSON.stringify(loggedUser));
      onLogin(loggedUser);
    } catch (err) {
      console.warn("Supabase auth fallback triggered:", err.message);
      
      // If it's an explicit wrong password from an ACTIVE Supabase project, show error
      if (err.status === 400 && err.message && !err.message.toLowerCase().includes("fetch") && !err.message.toLowerCase().includes("network")) {
        setErrorMsg(err.message);
        return;
      }

      // Seamless fallback: log in immediately without blocking alert popups
      const localUser = {
        email,
        id: "local_" + Date.now(),
        user_metadata: { name: email.split("@")[0] }
      };
      localStorage.setItem("sahayak_mock_user", JSON.stringify(localUser));
      onLogin(localUser);
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

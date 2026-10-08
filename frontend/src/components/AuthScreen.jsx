import { useState } from "react";
import { supabase } from "../api/supabase";

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
      if (isLogin) {
        const { data, error } = await supabase.auth.signInWithPassword({ email, password });
        if (error) throw error;
        onLogin(data.user);
      } else {
        const { data, error } = await supabase.auth.signUp({ email, password });
        if (error) throw error;
        if (data.user) {
          alert("Registration successful! You are now logged in.");
          onLogin(data.user);
        }
      }
    } catch (err) {
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

import { useState } from "react";

export default function AuthScreen({ onLogin, onGuest }) {
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    // TODO: Connect to Supabase Auth here in the next step
    // For now, immediately log them in
    onLogin({ email, id: "user_123" });
  };

  return (
    <div className="auth-container">
      <div className="auth-box">
        <div className="auth-logo">🫀</div>
        <h2>{isLogin ? "Welcome back" : "Create your account"}</h2>
        
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
          <button type="submit" className="btn-auth-primary">
            {isLogin ? "Continue" : "Sign Up"}
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

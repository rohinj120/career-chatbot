import ReactMarkdown from "react-markdown";
import { useState, useEffect, useRef } from "react";
import { io } from "socket.io-client";
import "./App.css";

const socket = io("http://localhost:8000", {
  transports: ["websocket"],
});

function Login({ onLogin }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    if (username.trim() && password.trim()) {
      onLogin();
    } else {
      setError("Please enter a username and password.");
    }
  };

  return (
    <div className="login-container">
      <div className="login-card">
        <h1>Career Chatbot</h1>
        <p className="login-subtitle">Sign in to continue</p>
        <form onSubmit={handleSubmit}>
          <div className="login-field">
            <label>Username</label>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="Enter username"
              autoFocus
            />
          </div>
          <div className="login-field">
            <label>Password</label>
            <div className="password-wrapper">
              <input
                type={showPassword ? "text" : "password"}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Enter password"
              />
              <button
                type="button"
                className="toggle-password"
                onClick={() => setShowPassword((prev) => !prev)}
              >
                {showPassword ? "Hide" : "Show"}
              </button>
            </div>
          </div>
          {error && <p className="login-error">{error}</p>}
          <button type="submit" className="login-btn">Login</button>
        </form>
      </div>
    </div>
  );
}

function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState([]);
  const [isTyping, setIsTyping] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const chatBoxRef = useRef(null);

  useEffect(() => {
    socket.on("connect", () => setIsConnected(true));
    socket.on("disconnect", () => setIsConnected(false));

    socket.on("typing", () => {
      setIsTyping(true);
      setMessages((prev) => [
        ...prev,
        { sender: "bot", text: "", sources: [] },
      ]);
    });

    socket.on("token", ({ token }) => {
      setMessages((prev) => {
        const updated = [...prev];
        const last = updated[updated.length - 1];
        if (last && last.sender === "bot") {
          updated[updated.length - 1] = { ...last, text: last.text + token };
        }
        return updated;
      });
    });

    socket.on("sources", ({ sources }) => {
      setMessages((prev) => {
        const updated = [...prev];
        const last = updated[updated.length - 1];
        if (last && last.sender === "bot") {
          updated[updated.length - 1] = { ...last, sources };
        }
        return updated;
      });
    });

    socket.on("done", () => setIsTyping(false));

    socket.on("error", ({ message: errMsg }) => {
      setIsTyping(false);
      setMessages((prev) => [
        ...prev,
        { sender: "bot", text: errMsg, sources: [] },
      ]);
    });

    return () => {
      socket.off("connect");
      socket.off("disconnect");
      socket.off("typing");
      socket.off("token");
      socket.off("sources");
      socket.off("done");
      socket.off("error");
    };
  }, []);

  useEffect(() => {
    if (chatBoxRef.current) {
      chatBoxRef.current.scrollTop = chatBoxRef.current.scrollHeight;
    }
  }, [messages]);

  const sendMessage = () => {
    if (!message.trim() || isTyping) return;
    const userText = message.trim();
    setMessage("");
    setMessages((prev) => [...prev, { sender: "user", text: userText }]);
    socket.emit("send_message", { message: userText });
  };

  if (!isLoggedIn) {
    return <Login onLogin={() => setIsLoggedIn(true)} />;
  }

  return (
    <div className="container">
      <div className="chat-header">
        <h1>Career Chatbot</h1>
        <button className="logout-btn" onClick={() => setIsLoggedIn(false)}>Logout</button>
      </div>

      {!isConnected && (
        <div className="connection-warning">Connecting to server...</div>
      )}

      <div className="chat-box" ref={chatBoxRef}>
        {messages.map((msg, index) => (
          <div
            key={index}
            className={msg.sender === "user" ? "user-message" : "bot-message"}
          >
            {msg.sender === "bot" ? <ReactMarkdown>{msg.text}</ReactMarkdown> : msg.text}

            {msg.sender === "bot" &&
              isTyping &&
              index === messages.length - 1 && (
                <span className="typing-cursor">▌</span>
              )}

            {msg.sender === "bot" && msg.sources && msg.sources.length > 0 && (
              <div className="sources">
                <span className="sources-label">Sources: </span>
                {msg.sources.map((s) => (
                  <span key={s} className="source-badge">
                    {s}
                  </span>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="input-area">
        <input
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder="Ask a career question..."
          disabled={isTyping}
          onKeyDown={(e) => {
            if (e.key === "Enter") sendMessage();
          }}
        />
        <button onClick={sendMessage} disabled={isTyping}>
          {isTyping ? "..." : "Send"}
        </button>
      </div>
    </div>
  );
}

export default App;

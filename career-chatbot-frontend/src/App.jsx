import ReactMarkdown from "react-markdown";
import { useState, useEffect, useRef } from "react";
import { io } from "socket.io-client";
import "./App.css";

const socket = io("http://localhost:8000", {
  transports: ["websocket"],
});

function App() {
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

  return (
    <div className="container">
      <h1>Career Chatbot</h1>

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

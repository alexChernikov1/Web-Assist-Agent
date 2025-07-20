import React, { useState, useEffect, useRef } from "react";
import "./ChatWindow.css";
import { getAIMessage } from "../api/api";
import { marked } from "marked";

function ChatWindow() {
  const [messages, setMessages] = useState([
    { role: "assistant", content: "Hi, how can I help you today?" },
  ]);
  const [input, setInput] = useState("");

  // remember where the placeholder lives so we can overwrite it
  const placeholderRef = useRef(null);
  const bottomRef = useRef(null);

  const scrollToBottom = () =>
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  useEffect(scrollToBottom, [messages]);

  /** exposed to api.js so it can push the progress message */
  const pushProgress = (msg) =>
    setMessages((prev) => {
      placeholderRef.current = prev.length;       // index of the placeholder
      return [...prev, msg];
    });

  const handleSend = async (text) => {
    const prompt = text.trim();
    if (!prompt) return;

    // add user message
    setMessages((prev) => [...prev, { role: "user", content: prompt }]);
    setInput("");

    // call API (may insert progress msg first)
    const reply = await getAIMessage(prompt, pushProgress);

    setMessages((prev) => {
      // if we showed a placeholder, overwrite it; otherwise append
      if (reply.progressShown && placeholderRef.current !== null) {
        return prev.map((m, i) =>
          i === placeholderRef.current ? { role: "assistant", content: reply.content } : m
        );
      }
      return [...prev, { role: "assistant", content: reply.content }];
    });

    placeholderRef.current = null; // reset for next round
  };

  return (
    <div className="messages-container">
      {messages.map((m, idx) => (
        <div key={idx} className={`${m.role}-message-container`}>
          <div className={`message ${m.role}-message`}>
            <div
              dangerouslySetInnerHTML={{
                __html: marked(m.content).replace(/<p>|<\/p>/g, ""),
              }}
            />
          </div>
        </div>
      ))}

      <div ref={bottomRef} />

      <div className="input-area">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type a message…"
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              handleSend(input);
            }
          }}
        />
        <button
          className="send-button"
          onClick={() => handleSend(input)}
        >
          Send
        </button>
      </div>
    </div>
  );
}

export default ChatWindow;



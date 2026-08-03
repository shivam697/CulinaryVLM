import { useState, useRef, useEffect } from 'react';
import { api } from '../api';

export default function AgentPage() {
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Hello! I\'m CulinaryVLM Agent 🍚 — I can answer complex questions about biryani cooking by using multiple tools. Try asking me to compare styles, find techniques, or explain recipes!', traces: [] },
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [sessionId] = useState(() => crypto.randomUUID());
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMsg = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userMsg }]);
    setLoading(true);

    try {
      const data = await api.agentQuery(userMsg, sessionId);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: data.answer,
        traces: data.tool_traces || [],
        plan: data.plan || '',
      }]);
    } catch (err) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `⚠️ ${err.message}\n\nThe agent layer may not be enabled. Set USE_AGENT_LAYER=true in .env and install requirements-agent.txt.`,
        traces: [],
      }]);
    } finally {
      setLoading(false);
    }
  };

  const suggestions = [
    'Compare Hyderabadi and Kolkata biryani dum methods',
    'What makes Malabar biryani unique?',
    'How many biryani styles use potatoes?',
    'Which style has the longest cooking time?',
  ];

  return (
    <div className="page-container" style={{ maxWidth: '850px', margin: '0 auto', padding: '1rem 1.5rem' }}>
      <div className="page-header animate-in">
        <h1>🤖 CulinaryVLM Agent</h1>
        <p>Multi-tool reasoning powered by LangGraph</p>
      </div>

      <div className="chat-container">
        {/* Messages */}
        <div className="chat-messages">
          {messages.map((msg, i) => (
            <div key={i} className={`chat-bubble ${msg.role}`}>
              <div style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</div>

              {/* Tool traces */}
              {msg.traces?.length > 0 && (
                <div className="tool-trace">
                  <div style={{ fontWeight: 600, marginBottom: '0.5rem', fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                    🔧 Tools Used
                  </div>
                  {msg.traces.map((t, j) => (
                    <div key={j} className="tool-trace-item">
                      <span className="tool-name">{t.tool_name}</span>
                      <span className="tool-duration">{t.duration_ms?.toFixed(0)}ms</span>
                    </div>
                  ))}
                </div>
              )}

              {msg.plan && (
                <div style={{ marginTop: '0.5rem', fontSize: '0.8rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>
                  Plan: {msg.plan}
                </div>
              )}
            </div>
          ))}

          {loading && (
            <div className="chat-bubble assistant" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <div className="spinner" style={{ width: '20px', height: '20px', borderWidth: '2px' }} />
              <span style={{ color: 'var(--text-muted)' }}>Thinking...</span>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Suggestions (show only at start) */}
        {messages.length <= 1 && (
          <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', padding: '0.5rem 0' }}>
            {suggestions.map(q => (
              <button key={q} className="btn btn-secondary" style={{ fontSize: '0.8rem' }}
                onClick={() => setInput(q)}>
                {q}
              </button>
            ))}
          </div>
        )}

        {/* Input */}
        <form onSubmit={handleSend} className="chat-input-area">
          <input
            className="input"
            placeholder="Ask the agent anything about biryani..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={loading}
          />
          <button type="submit" className="btn btn-primary" disabled={loading || !input.trim()}>
            Send
          </button>
        </form>
      </div>
    </div>
  );
}

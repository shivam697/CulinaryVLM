import { useState, useRef, useEffect, useMemo } from 'react';
import { api } from '../api';
import { CATEGORY_EMOJIS } from '../constants/biryaniStyles';
import ChatSidebar from '../components/ChatSidebar';
import ChatBubble from '../components/ChatBubble';
import ChatInput from '../components/ChatInput';
import SuggestionChips from '../components/SuggestionChips';
import Hyperspeed from '../components/Hyperspeed';

export default function AgentPage() {
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Hello! I\'m CulinaryVLM Agent 🍚 — I can answer complex questions about biryani cooking by using multiple tools. Try asking me to compare styles, find techniques, or explain recipes!', traces: [] },
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [sessionId] = useState(() => crypto.randomUUID());
  const [activeStyle, setActiveStyle] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
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
      // Prepend style context if a specific style is selected
      const contextMsg = activeStyle
        ? `[Context: ${activeStyle} biryani] ${userMsg}`
        : userMsg;
      const data = await api.agentQuery(contextMsg, sessionId);
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

  const suggestions = activeStyle
    ? [
        `What spices are used in ${activeStyle} biryani?`,
        `Describe the dum method for ${activeStyle} biryani`,
        `What makes ${activeStyle} biryani unique?`,
        `How is ${activeStyle} biryani traditionally served?`,
      ]
    : [
        'Compare Hyderabadi and Kolkata biryani dum methods',
        'What makes Malabar biryani unique?',
        'How many biryani styles use potatoes?',
        'Which style has the longest cooking time?',
      ];

  const hyperspeedOptions = useMemo(() => ({
    distortion: 'turbulentDistortion',
    length: 400,
    roadWidth: 10,
    islandWidth: 2,
    lanesPerRoad: 4,
    fov: 90,
    fovSpeedUp: 150,
    speedUp: 2,
    carLightsFade: 0.4,
    totalSideLightSticks: 20,
    lightPairsPerRoadWay: 40,
    shoulderLinesWidthPercentage: 0.05,
    brokenLinesWidthPercentage: 0.1,
    brokenLinesLengthPercentage: 0.5,
    lightStickWidth: [0.12, 0.5],
    lightStickHeight: [1.3, 1.7],
    movingAwaySpeed: [60, 80],
    movingCloserSpeed: [-120, -160],
    carLightsLength: [400 * 0.03, 400 * 0.2],
    carLightsRadius: [0.05, 0.14],
    carWidthPercentage: [0.3, 0.5],
    carShiftX: [-0.8, 0.8],
    carFloorSeparation: [0, 5],
    colors: {
      roadColor: 0x080808,
      islandColor: 0x0a0a0a,
      background: 0x000000,
      shoulderLines: 0xFFFFFF,
      brokenLines: 0xFFFFFF,
      leftCars: [0xD856BF, 0x6750A2, 0xC247AC],
      rightCars: [0x03B3C3, 0x0E5EA5, 0x324555],
      sticks: 0x03B3C3,
    }
  }), []);

  return (
    <div className={`agent-layout${sidebarOpen ? ' sidebar-open' : ''}`}>
      {/* Hyperspeed animated background */}
      <div className="agent-background">
        <Hyperspeed effectOptions={hyperspeedOptions} />
      </div>

      {/* Mobile sidebar toggle */}
      <button
        className="btn btn-ghost sidebar-toggle"
        onClick={() => setSidebarOpen(!sidebarOpen)}
        aria-label="Toggle biryani styles sidebar"
      >
        ☰
      </button>

      {/* Sidebar */}
      <ChatSidebar
        activeStyle={activeStyle}
        onStyleSelect={(style) => {
          setActiveStyle(style);
          setSidebarOpen(false);
        }}
      />

      {/* Sidebar backdrop for mobile */}
      {sidebarOpen && (
        <div
          className="sidebar-backdrop"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Main chat area */}
      <div className="agent-main">
        {/* Chat header */}
        <div className="chat-header">
          <div className="chat-header-info">
            <span className="chat-header-emoji">
              {activeStyle ? (CATEGORY_EMOJIS[activeStyle] || '🍚') : '🤖'}
            </span>
            <div>
              <h2 className="chat-header-title">
                {activeStyle ? `${activeStyle} Biryani` : 'CulinaryVLM Agent'}
              </h2>
              <p className="chat-header-subtitle">
                {activeStyle
                  ? `Focused on ${activeStyle} regional style`
                  : 'Multi-tool reasoning powered by LangGraph'}
              </p>
            </div>
          </div>
        </div>

        <div className="chat-container">
          {/* Messages */}
          <div className="chat-messages" role="log" aria-live="polite">
            {messages.map((msg, i) => (
              <ChatBubble key={i} message={msg} index={i} />
            ))}

            {loading && (
              <div className="typing-indicator">
                <div className="typing-dots">
                  <span /><span /><span />
                </div>
                <span style={{ color: 'var(--text-muted)', fontSize: 'var(--text-sm)' }}>Thinking...</span>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Suggestions (show only at start) */}
          {messages.length <= 1 && (
            <SuggestionChips
              suggestions={suggestions}
              onSelect={(q) => setInput(q)}
            />
          )}

          {/* Input */}
          <ChatInput
            value={input}
            onChange={setInput}
            onSubmit={handleSend}
            loading={loading}
            placeholder={activeStyle
              ? `Ask about ${activeStyle} biryani...`
              : 'Ask the agent anything about biryani...'}
          />
        </div>
      </div>
    </div>
  );
}

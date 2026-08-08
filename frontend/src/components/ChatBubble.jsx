export default function ChatBubble({ message }) {
  const { role, content, traces, plan } = message;

  return (
    <div className={`chat-bubble ${role}`}>
      <div className="chat-bubble-sender">
        {role === 'user' ? '👤 You' : '🍚 CulinaryVLM'}
      </div>

      <div className="chat-bubble-content">{content}</div>

      {/* Tool traces */}
      {traces?.length > 0 && (
        <div className="tool-trace">
          <div className="tool-trace-header">
            🔧 Tools Used
          </div>
          {traces.map((t, j) => (
            <div key={j} className="tool-trace-item">
              <span className="tool-name">{t.tool_name}</span>
              <span className="tool-duration">{t.duration_ms?.toFixed(0)}ms</span>
            </div>
          ))}
        </div>
      )}

      {plan && (
        <div className="chat-bubble-plan">
          📋 Plan: {plan}
        </div>
      )}
    </div>
  );
}

import { motion } from 'motion/react';

/**
 * ChatBubble — renders a single chat message with sender label, content,
 * tool traces, and plan. Uses motion for fade+slide entrance animation.
 *
 * Props:
 *   message: { role: string, content: string, traces?: array, plan?: string }
 *   index: number (used for stagger delay)
 */
export default function ChatBubble({ message, index = 0 }) {
  const { role, content, traces, plan } = message;

  return (
    <motion.div
      className={`chat-bubble ${role}`}
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{
        duration: 0.3,
        delay: Math.min(index * 0.05, 0.3),
        ease: [0.25, 0.46, 0.45, 0.94],
      }}
    >
      <div className="chat-bubble-sender">
        {role === 'user' ? '👤 You' : '🍚 CulinaryVLM'}
      </div>

      <div className="chat-bubble-content">{content}</div>

      {/* Tool traces */}
      {traces?.length > 0 && (
        <motion.div
          className="tool-trace"
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          transition={{ duration: 0.25, delay: 0.15 }}
        >
          <div className="tool-trace-header">
            🔧 Tools Used
          </div>
          {traces.map((t, j) => (
            <div key={j} className="tool-trace-item">
              <span className="tool-name">{t.tool_name}</span>
              <span className="tool-duration">{t.duration_ms?.toFixed(0)}ms</span>
            </div>
          ))}
        </motion.div>
      )}

      {plan && (
        <div className="chat-bubble-plan">
          📋 Plan: {plan}
        </div>
      )}
    </motion.div>
  );
}

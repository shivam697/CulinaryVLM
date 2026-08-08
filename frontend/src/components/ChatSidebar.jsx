import { motion } from 'motion/react';
import { CATEGORY_EMOJIS } from '../constants/biryaniStyles';

/** The 5 primary styles featured in the chat sidebar. */
const FEATURED_STYLES = ['Hyderabadi', 'Kolkata', 'Lucknowi', 'Malabar', 'Sindhi'];

/**
 * ChatSidebar — sidebar listing biryani regional styles with smooth
 * stagger animation on mount.
 *
 * Props:
 *   activeStyle: string | null
 *   onStyleSelect: (style: string | null) => void
 */
export default function ChatSidebar({ activeStyle, onStyleSelect }) {
  const allStyles = [{ name: null, label: 'All Styles', emoji: '🍚' },
    ...FEATURED_STYLES.map(s => ({ name: s, label: s, emoji: CATEGORY_EMOJIS[s] || '🍚' })),
  ];

  return (
    <aside className="chat-sidebar" aria-label="Biryani style selector">
      <div className="chat-sidebar-header">
        <h3>Biryani Styles</h3>
        <p>Select a regional focus</p>
      </div>

      <nav className="chat-sidebar-nav">
        {allStyles.map((item, i) => (
          <motion.button
            key={item.label}
            className={`chat-sidebar-item ${activeStyle === item.name ? 'active' : ''}`}
            onClick={() => onStyleSelect(item.name)}
            aria-current={activeStyle === item.name ? 'true' : undefined}
            initial={{ opacity: 0, x: -16 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{
              duration: 0.25,
              delay: i * 0.06,
              ease: [0.25, 0.46, 0.45, 0.94],
            }}
            whileHover={{ x: 4 }}
            whileTap={{ scale: 0.97 }}
          >
            <span className="chat-sidebar-emoji">{item.emoji}</span>
            <span className="chat-sidebar-label">{item.label}</span>
          </motion.button>
        ))}
      </nav>

      <div className="chat-sidebar-footer">
        <p>Powered by CulinaryVLM</p>
      </div>
    </aside>
  );
}

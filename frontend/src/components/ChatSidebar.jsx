import { BIRYANI_STYLES, CATEGORY_EMOJIS } from '../constants/biryaniStyles';

/** The 5 primary styles featured in the chat sidebar. */
const FEATURED_STYLES = ['Hyderabadi', 'Kolkata', 'Lucknowi', 'Malabar', 'Sindhi'];

export default function ChatSidebar({ activeStyle, onStyleSelect }) {
  return (
    <aside className="chat-sidebar" aria-label="Biryani style selector">
      <div className="chat-sidebar-header">
        <h3>Biryani Styles</h3>
        <p>Select a regional focus</p>
      </div>

      <nav className="chat-sidebar-nav">
        {/* "All Styles" option */}
        <button
          className={`chat-sidebar-item ${!activeStyle ? 'active' : ''}`}
          onClick={() => onStyleSelect(null)}
          aria-current={!activeStyle ? 'true' : undefined}
        >
          <span className="chat-sidebar-emoji">🍚</span>
          <span className="chat-sidebar-label">All Styles</span>
        </button>

        {/* Featured styles */}
        {FEATURED_STYLES.map((style) => (
          <button
            key={style}
            className={`chat-sidebar-item ${activeStyle === style ? 'active' : ''}`}
            onClick={() => onStyleSelect(style)}
            aria-current={activeStyle === style ? 'true' : undefined}
          >
            <span className="chat-sidebar-emoji">{CATEGORY_EMOJIS[style] || '🍚'}</span>
            <span className="chat-sidebar-label">{style}</span>
          </button>
        ))}
      </nav>

      <div className="chat-sidebar-footer">
        <p>Powered by CulinaryVLM</p>
      </div>
    </aside>
  );
}

export default function ChatInput({ value, onChange, onSubmit, loading, placeholder }) {
  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit(e);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (value.trim() && !loading) {
        onSubmit(e);
      }
    }
  };

  return (
    <form onSubmit={handleSubmit} className="chat-input-area">
      <input
        className="input"
        placeholder={placeholder || 'Type a message...'}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={handleKeyDown}
        disabled={loading}
        aria-label="Chat message input"
      />
      <button
        type="submit"
        className="btn btn-primary"
        disabled={loading || !value.trim()}
        aria-label="Send message"
      >
        {loading ? '⏳' : '↑'} Send
      </button>
    </form>
  );
}

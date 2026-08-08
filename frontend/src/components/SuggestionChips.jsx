export default function SuggestionChips({ suggestions, onSelect }) {
  return (
    <div className="suggestion-chips">
      {suggestions.map((q) => (
        <button
          key={q}
          className="btn btn-secondary suggestion-chip"
          onClick={() => onSelect(q)}
        >
          {q}
        </button>
      ))}
    </div>
  );
}

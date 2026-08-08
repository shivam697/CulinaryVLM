import { motion } from 'motion/react';

/**
 * SuggestionChips — horizontal list of clickable suggestion buttons
 * with stagger fade-in animation.
 *
 * Props:
 *   suggestions: string[]
 *   onSelect: (suggestion: string) => void
 */
export default function SuggestionChips({ suggestions, onSelect }) {
  return (
    <div className="suggestion-chips">
      {suggestions.map((q, i) => (
        <motion.button
          key={q}
          className="btn btn-secondary suggestion-chip"
          onClick={() => onSelect(q)}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{
            duration: 0.25,
            delay: i * 0.07,
            ease: [0.25, 0.46, 0.45, 0.94],
          }}
          whileHover={{ scale: 1.03 }}
          whileTap={{ scale: 0.97 }}
        >
          {q}
        </motion.button>
      ))}
    </div>
  );
}

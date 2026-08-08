import { motion } from 'motion/react';

/**
 * StatCard — animated stat display card with stagger support.
 *
 * Props:
 *   value: string | number
 *   label: string
 *   index?: number (for stagger delay)
 */
export default function StatCard({ value, label, index = 0 }) {
  return (
    <motion.div
      className="stat-card"
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{
        duration: 0.35,
        delay: index * 0.08,
        ease: [0.25, 0.46, 0.45, 0.94],
      }}
    >
      <span className="stat-value">{value}</span>
      <span className="stat-label">{label}</span>
    </motion.div>
  );
}

import { motion } from 'motion/react';

/**
 * PageHeader — animated page title + subtitle used on all pages.
 *
 * Props:
 *   title: string
 *   subtitle?: string
 */
export default function PageHeader({ title, subtitle }) {
  return (
    <motion.div
      className="page-header"
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, ease: [0.25, 0.46, 0.45, 0.94] }}
    >
      <h1>{title}</h1>
      {subtitle && <p>{subtitle}</p>}
    </motion.div>
  );
}

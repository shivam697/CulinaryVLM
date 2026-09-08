import { useState, useEffect } from 'react';
import FloatingLines from './FloatingLines';

/**
 * SafeFloatingLines — accessibility + performance wrapper around the vendored
 * FloatingLines component. Follows the same pattern as SafePrism.
 *
 * Handles:
 *  1. prefers-reduced-motion: don't mount FloatingLines at all — fall back to
 *     a static panel. Unlike Prism (which has a timeScale=0 freeze mode),
 *     FloatingLines has no built-in freeze — its rAF loop drives iTime
 *     unconditionally, so the only safe option is to not render it.
 *  2. Low-end device detection: same heuristic as SafePrism (≤2 cores or
 *     saveData) → skip WebGL entirely.
 */
export default function SafeFloatingLines(props) {
  const [prefersReduced, setPrefersReduced] = useState(() =>
    typeof window !== 'undefined' &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches
  );

  const [isLowEnd] = useState(() => {
    if (typeof window === 'undefined') return false;
    if (navigator.hardwareConcurrency && navigator.hardwareConcurrency <= 2) return true;
    if (navigator.connection?.saveData) return true;
    return false;
  });

  useEffect(() => {
    const mql = window.matchMedia('(prefers-reduced-motion: reduce)');
    const onChange = (e) => setPrefersReduced(e.matches);
    mql.addEventListener('change', onChange);
    return () => mql.removeEventListener('change', onChange);
  }, []);

  // Reduced motion or low-end: render nothing (the parent's background
  // color from --bg-primary shows through, which is the correct fallback)
  if (prefersReduced || isLowEnd) {
    return null;
  }

  return <FloatingLines {...props} />;
}

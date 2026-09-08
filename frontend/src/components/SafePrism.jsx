import { useState, useEffect } from 'react';
import Prism from './Prism';

/**
 * SafePrism — accessibility + performance wrapper around the vendored Prism
 * component. This file is the one consumers should import, not Prism.jsx
 * directly.
 *
 * Handles:
 *  1. prefers-reduced-motion: freeze animation (timeScale=0) so the geometry
 *     is still visible but static — matches the rest of the site's
 *     @media (prefers-reduced-motion: reduce) posture.
 *  2. Low-end device detection: falls back to a static CSS panel on devices
 *     with ≤2 logical cores, or where the user is on a data-saver connection.
 *     A 100-step raymarch per frame can still be janky on low-end mobile GPUs,
 *     even with dpr capped at 2.
 */
export default function SafePrism(props) {
  const [prefersReduced, setPrefersReduced] = useState(() =>
    typeof window !== 'undefined' &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches
  );

  const [isLowEnd] = useState(() => {
    if (typeof window === 'undefined') return false;
    // ≤2 cores → very likely a low-end mobile device
    if (navigator.hardwareConcurrency && navigator.hardwareConcurrency <= 2) return true;
    // Save-Data header respected by some browsers
    if (navigator.connection?.saveData) return true;
    return false;
  });

  // Listen for changes (user can toggle the preference live in system settings)
  useEffect(() => {
    const mql = window.matchMedia('(prefers-reduced-motion: reduce)');
    const onChange = (e) => setPrefersReduced(e.matches);
    mql.addEventListener('change', onChange);
    return () => mql.removeEventListener('change', onChange);
  }, []);

  // Low-end: don't mount Prism at all → static fallback panel
  if (isLowEnd) {
    return (
      <div
        className="prism-fallback"
        style={{
          position: 'absolute',
          inset: 0,
          background: 'var(--bg-secondary)',
        }}
      />
    );
  }

  // Reduced motion: mount Prism but freeze time so geometry is visible but static
  const overrides = prefersReduced ? { timeScale: 0 } : {};

  return <Prism {...props} {...overrides} />;
}

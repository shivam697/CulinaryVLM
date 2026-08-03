import { useState, useEffect } from 'react';
import { api } from '../api';

const STYLES = [
  'Ambur', 'Andhra', 'Arabic', 'Assamese', 'Bamboo', 'Bihari', 'Bombay',
  'Degi', 'Delhi', 'Dindigul', 'Hyderabadi', 'Kashmiri', 'Kolkata',
  'Lucknowi', 'Malabar', 'Matka', 'Mughlai', 'Muradabadi', 'Sindhi', 'Tandoori',
];

export default function ComparePage() {
  const [catA, setCatA] = useState('Hyderabadi');
  const [catB, setCatB] = useState('Kolkata');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleCompare = async () => {
    if (catA === catB) return;
    setLoading(true);
    try {
      const data = await api.compare(catA, catB);
      setResult(data);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { handleCompare(); }, []);

  return (
    <div className="page-container">
      <div className="page-header animate-in">
        <h1>Compare Biryani Styles</h1>
        <p>Side-by-side comparison of regional biryani cooking methods</p>
      </div>

      {/* Selectors */}
      <div style={{ display: 'flex', gap: '1rem', alignItems: 'center', marginBottom: '2rem', flexWrap: 'wrap' }}>
        <select className="input" value={catA} onChange={(e) => setCatA(e.target.value)}
          style={{ maxWidth: '220px' }}>
          {STYLES.map(s => <option key={s} value={s}>{s}</option>)}
        </select>

        <span style={{ fontFamily: 'var(--font-display)', fontWeight: 800, color: 'var(--text-muted)', fontSize: '1.25rem' }}>
          VS
        </span>

        <select className="input" value={catB} onChange={(e) => setCatB(e.target.value)}
          style={{ maxWidth: '220px' }}>
          {STYLES.map(s => <option key={s} value={s}>{s}</option>)}
        </select>

        <button className="btn btn-primary" onClick={handleCompare} disabled={loading || catA === catB}>
          {loading ? '⟳' : '⚡'} Compare
        </button>
      </div>

      {/* Results */}
      {result && (
        <div className="animate-in">
          {/* Summary */}
          <div className="card-glass" style={{ marginBottom: '1.5rem', padding: '1.25rem' }}>
            <p style={{ fontSize: '0.95rem', lineHeight: 1.7 }}>{result.summary}</p>
          </div>

          {/* Difference rows */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            {result.differences?.map((diff, i) => (
              <div key={i} className="diff-row animate-slide-up" style={{ animationDelay: `${i * 60}ms` }}>
                <div className="diff-label">{diff.aspect}</div>
                <div className="diff-value">
                  <div style={{ fontSize: '0.75rem', color: 'var(--accent-primary)', marginBottom: '0.25rem', fontWeight: 600 }}>
                    {result.category_a}
                  </div>
                  {diff.category_a_value}
                </div>
                <div className="diff-value">
                  <div style={{ fontSize: '0.75rem', color: 'var(--accent-blue)', marginBottom: '0.25rem', fontWeight: 600 }}>
                    {result.category_b}
                  </div>
                  {diff.category_b_value}
                </div>
                {diff.explanation && (
                  <p style={{ gridColumn: '1 / -1', fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
                    {diff.explanation}
                  </p>
                )}
              </div>
            ))}
          </div>

          {result.differences?.length === 0 && (
            <div className="empty-state">
              <div className="empty-icon">🤝</div>
              <p>These styles are quite similar!</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

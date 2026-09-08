import { useState, useEffect } from 'react';
import { api } from '../api';
import { BIRYANI_STYLES } from '../constants/biryaniStyles';
import PageHeader from '../components/PageHeader';
import SoftAurora from '../components/SoftAurora';

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
    <div className="compare-layout">
      {/* SoftAurora animated background */}
      <div className="compare-background">
        <SoftAurora
          speed={0.6}
          scale={1.5}
          brightness={1.0}
          color1="#f7f7f7"
          color2="#e100ff"
          noiseFrequency={2.5}
          noiseAmplitude={1.0}
          bandHeight={0.5}
          bandSpread={1.0}
          octaveDecay={0.1}
          layerOffset={0}
          colorSpeed={1.0}
          enableMouseInteraction={true}
          mouseInfluence={0.25}
        />
      </div>

      {/* Page content */}
      <div className="compare-content">
        <PageHeader
          title="Compare Biryani Styles"
          subtitle="Side-by-side comparison of regional biryani cooking methods"
        />

        {/* Selectors */}
        <div style={{ display: 'flex', gap: 'var(--space-4)', alignItems: 'center', marginBottom: 'var(--space-8)', flexWrap: 'wrap' }}>
          <select className="input" value={catA} onChange={(e) => setCatA(e.target.value)}
            aria-label="First biryani style to compare"
            style={{ maxWidth: '220px' }}>
            {BIRYANI_STYLES.map(s => <option key={s} value={s}>{s}</option>)}
          </select>

          <span style={{ fontFamily: 'var(--font-display)', fontWeight: 800, color: 'var(--text-muted)', fontSize: 'var(--text-xl)' }}>
            VS
          </span>

          <select className="input" value={catB} onChange={(e) => setCatB(e.target.value)}
            aria-label="Second biryani style to compare"
            style={{ maxWidth: '220px' }}>
            {BIRYANI_STYLES.map(s => <option key={s} value={s}>{s}</option>)}
          </select>

          <button className="btn btn-primary" onClick={handleCompare} disabled={loading || catA === catB}>
            {loading ? '⟳' : '⚡'} Compare
          </button>
        </div>

        {/* Results */}
        {result && (
          <div className="animate-in">
            {/* Summary */}
            <div className="card-glass" style={{ marginBottom: 'var(--space-6)', padding: 'var(--space-5)' }}>
              <p style={{ fontSize: 'var(--text-base)', lineHeight: 1.7 }}>{result.summary}</p>
            </div>

            {/* Difference rows */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-3)' }}>
              {result.differences?.map((diff, i) => (
                <div key={i} className="diff-row animate-slide-up" style={{ animationDelay: `${i * 60}ms` }}>
                  <div className="diff-label">{diff.aspect}</div>
                  <div className="diff-value">
                    <div style={{ fontSize: 'var(--text-xs)', color: 'var(--accent-primary)', marginBottom: 'var(--space-1)', fontWeight: 600 }}>
                      {result.category_a}
                    </div>
                    {diff.category_a_value}
                  </div>
                  <div className="diff-value">
                    <div style={{ fontSize: 'var(--text-xs)', color: 'var(--accent-blue)', marginBottom: 'var(--space-1)', fontWeight: 600 }}>
                      {result.category_b}
                    </div>
                    {diff.category_b_value}
                  </div>
                  {diff.explanation && (
                    <p style={{ gridColumn: '1 / -1', fontSize: 'var(--text-sm)', color: 'var(--text-muted)', marginTop: 'var(--space-1)' }}>
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
    </div>
  );
}

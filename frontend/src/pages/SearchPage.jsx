import { useState } from 'react';
import { api } from '../api';

export default function SearchPage() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;
    setLoading(true);
    setError('');
    try {
      const data = await api.search(query.trim());
      setResults(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page-container">
      <div className="page-header animate-in">
        <h1>Search Cooking Techniques</h1>
        <p>Find specific cooking steps across hundreds of biryani videos using semantic search</p>
      </div>

      <form onSubmit={handleSearch} style={{ marginBottom: '2rem' }}>
        <div className="input-group" style={{ maxWidth: '700px' }}>
          <input
            className="input input-lg"
            type="text"
            placeholder="e.g., marinating chicken with yogurt and spices..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            style={{ borderRadius: 'var(--radius-lg) 0 0 var(--radius-lg)' }}
          />
          <button
            type="submit"
            className="btn btn-primary"
            disabled={loading}
            style={{ borderRadius: '0 var(--radius-lg) var(--radius-lg) 0', padding: '0 2rem' }}
          >
            {loading ? '⟳' : '🔍'} Search
          </button>
        </div>
      </form>

      {error && (
        <div className="card" style={{ borderColor: 'var(--accent-secondary)', marginBottom: '1.5rem' }}>
          <p style={{ color: 'var(--accent-secondary)' }}>⚠️ {error}</p>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginTop: '0.5rem' }}>
            Note: Semantic search requires the FAISS index. Run Stage 11 first.
          </p>
        </div>
      )}

      {results && results.total_results === 0 && (
        <div className="empty-state">
          <div className="empty-icon">🔍</div>
          <p>No segments found for "{results.query}"</p>
          <p style={{ fontSize: '0.85rem', marginTop: '0.5rem' }}>
            Try different keywords like "frying onions", "layering rice", or "dum cooking"
          </p>
        </div>
      )}

      {results && results.total_results > 0 && (
        <div className="animate-in">
          <p style={{ color: 'var(--text-secondary)', marginBottom: '1rem', fontSize: '0.9rem' }}>
            Found <strong style={{ color: 'var(--accent-primary)' }}>{results.total_results}</strong> segments
            for "{results.query}"
          </p>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            {results.results.map((r, i) => (
              <div key={i} className="card" style={{ display: 'flex', gap: '1rem', alignItems: 'flex-start' }}>
                <div style={{
                  minWidth: '48px', height: '48px',
                  borderRadius: 'var(--radius-sm)',
                  background: 'var(--gradient-primary)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontWeight: 800, fontSize: '1.1rem', color: '#fff',
                }}>
                  {Math.round(r.score * 100)}
                </div>
                <div style={{ flex: 1 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
                    <h4>{r.action}</h4>
                    <span className="badge badge-saffron">{r.category}</span>
                  </div>
                  <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>{r.description}</p>
                  <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.5rem' }}>
                    ⏱ {r.start_time?.toFixed(1)}s — {r.end_time?.toFixed(1)}s
                    {r.video_title && ` • ${r.video_title}`}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {!results && !loading && (
        <div style={{ marginTop: '2rem' }}>
          <h3 style={{ marginBottom: '1rem', color: 'var(--text-secondary)' }}>Try searching for:</h3>
          <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
            {['marinating chicken', 'frying onions golden', 'layering rice and meat', 'dum cooking sealed', 'saffron milk drizzle'].map((q) => (
              <button key={q} className="btn btn-secondary" onClick={() => { setQuery(q); }}>
                {q}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

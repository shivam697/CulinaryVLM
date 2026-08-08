import { useState } from 'react';
import { api } from '../api';
import PageHeader from '../components/PageHeader';
import SuggestionChips from '../components/SuggestionChips';

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
      <PageHeader
        title="Search Cooking Techniques"
        subtitle="Find specific cooking steps across hundreds of biryani videos using semantic search"
      />

      <form onSubmit={handleSearch} style={{ marginBottom: 'var(--space-8)' }}>
        <div className="input-group" style={{ maxWidth: 'var(--container-narrow)' }}>
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
            style={{ borderRadius: '0 var(--radius-lg) var(--radius-lg) 0', padding: '0 var(--space-8)' }}
          >
            {loading ? '⟳' : '🔍'} Search
          </button>
        </div>
      </form>

      {error && (
        <div className="card" style={{ borderColor: 'var(--accent-secondary)', marginBottom: 'var(--space-6)' }}>
          <p style={{ color: 'var(--accent-secondary)' }}>⚠️ {error}</p>
          <p style={{ color: 'var(--text-muted)', fontSize: 'var(--text-sm)', marginTop: 'var(--space-2)' }}>
            Note: Semantic search requires the FAISS index. Run Stage 11 first.
          </p>
        </div>
      )}

      {results && results.total_results === 0 && (
        <div className="empty-state">
          <div className="empty-icon">🔍</div>
          <p>No segments found for "{results.query}"</p>
          <p style={{ fontSize: 'var(--text-sm)', marginTop: 'var(--space-2)' }}>
            Try different keywords like "frying onions", "layering rice", or "dum cooking"
          </p>
        </div>
      )}

      {results && results.total_results > 0 && (
        <div className="animate-in">
          <p style={{ color: 'var(--text-secondary)', marginBottom: 'var(--space-4)', fontSize: 'var(--text-base)' }}>
            Found <strong style={{ color: 'var(--accent-primary)' }}>{results.total_results}</strong> segments
            for "{results.query}"
          </p>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-3)' }}>
            {results.results.map((r, i) => (
              <div key={i} className="card" style={{ display: 'flex', gap: 'var(--space-4)', alignItems: 'flex-start' }}>
                <div style={{
                  minWidth: '48px', height: '48px',
                  borderRadius: 'var(--radius-sm)',
                  background: 'var(--gradient-primary)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontWeight: 800, fontSize: 'var(--text-lg)', color: '#fff',
                }}>
                  {Math.round(r.score * 100)}
                </div>
                <div style={{ flex: 1 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)', marginBottom: 'var(--space-2)' }}>
                    <h4>{r.action}</h4>
                    <span className="badge badge-saffron">{r.category}</span>
                  </div>
                  <p style={{ fontSize: 'var(--text-base)', color: 'var(--text-secondary)' }}>{r.description}</p>
                  <p style={{ fontSize: 'var(--text-sm)', color: 'var(--text-muted)', marginTop: 'var(--space-2)' }}>
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
        <div style={{ marginTop: 'var(--space-8)' }}>
          <h3 style={{ marginBottom: 'var(--space-4)', color: 'var(--text-secondary)' }}>Try searching for:</h3>
          <SuggestionChips
            suggestions={['marinating chicken', 'frying onions golden', 'layering rice and meat', 'dum cooking sealed', 'saffron milk drizzle']}
            onSelect={(q) => setQuery(q)}
          />
        </div>
      )}
    </div>
  );
}

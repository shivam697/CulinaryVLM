import { useState, useEffect } from 'react';
import { api } from '../api';

const CATEGORY_EMOJIS = {
  Hyderabadi: '🌶️', Kolkata: '🥚', Lucknowi: '🥀', Malabar: '🥥',
  Sindhi: '🧅', Muradabadi: '🍖', Delhi: '🏛️', Andhra: '🔥',
  Bihari: '🫕', Assamese: '🎋', Bombay: '🌆', Kashmiri: '🏔️',
  Mughlai: '👑', Bamboo: '🎍', Degi: '🫕', Matka: '🏺',
  Tandoori: '🔥', Arabic: '🕌', Ambur: '🌿', Dindigul: '🌡️',
  Generic: '🍚',
};

export default function ExplorePage() {
  const [recipes, setRecipes] = useState([]);
  const [stats, setStats] = useState(null);
  const [selectedRecipe, setSelectedRecipe] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([api.getRecipes(), api.getVideoStats()])
      .then(([r, s]) => { setRecipes(r.recipes || []); setStats(s); })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <div className="loading-container"><div className="spinner" /><p>Loading recipes...</p></div>;
  }

  return (
    <div className="page-container">
      <div className="page-header animate-in">
        <h1>Explore Biryani Styles</h1>
        <p>Discover 20 regional styles of chicken biryani across India and beyond</p>
      </div>

      {/* Stats */}
      {stats && (
        <div className="grid grid-4 stagger" style={{ marginBottom: '2rem' }}>
          <div className="stat-card animate-slide-up">
            <span className="stat-value">{stats.total_videos}</span>
            <span className="stat-label">Videos Analyzed</span>
          </div>
          <div className="stat-card animate-slide-up">
            <span className="stat-value">{Object.keys(stats.categories || {}).length}</span>
            <span className="stat-label">Regional Styles</span>
          </div>
          <div className="stat-card animate-slide-up">
            <span className="stat-value">{Object.keys(stats.languages || {}).length}</span>
            <span className="stat-label">Languages</span>
          </div>
          <div className="stat-card animate-slide-up">
            <span className="stat-value">{recipes.length}</span>
            <span className="stat-label">Canonical Recipes</span>
          </div>
        </div>
      )}

      {/* Recipe Grid */}
      <div className="grid grid-3 stagger">
        {recipes.map((recipe) => (
          <div
            key={recipe.category}
            className="card animate-slide-up"
            style={{ cursor: 'pointer' }}
            onClick={() => setSelectedRecipe(selectedRecipe?.category === recipe.category ? null : recipe)}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.75rem' }}>
              <span style={{ fontSize: '1.75rem' }}>{CATEGORY_EMOJIS[recipe.category] || '🍚'}</span>
              <div>
                <h3 style={{ fontSize: '1.1rem' }}>{recipe.category}</h3>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{recipe.region}</span>
              </div>
            </div>

            <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '0.75rem' }}>
              <span className="badge badge-saffron">{recipe.steps?.length || 0} steps</span>
              <span className="badge badge-blue">{recipe.rice_type?.split(' ')[0]}</span>
              {recipe.dum_method && <span className="badge badge-purple">{recipe.dum_method.split(' ')[0]} dum</span>}
            </div>

            {recipe.distinguishing_features?.slice(0, 2).map((f, i) => (
              <p key={i} style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '0.25rem' }}>
                • {f.length > 80 ? f.slice(0, 80) + '…' : f}
              </p>
            ))}

            {/* Expanded Detail */}
            {selectedRecipe?.category === recipe.category && (
              <div style={{ marginTop: '1rem', paddingTop: '1rem', borderTop: '1px solid var(--border-subtle)' }}>
                <h4 style={{ color: 'var(--accent-primary)', marginBottom: '0.75rem' }}>Key Steps</h4>
                {recipe.steps?.filter(s => s.is_defining_step).map((step) => (
                  <div key={step.step_number} style={{ marginBottom: '0.5rem', fontSize: '0.85rem' }}>
                    <span style={{ color: 'var(--accent-primary)', fontWeight: 600 }}>{step.step_number}.</span>{' '}
                    <strong>{step.action}</strong> — {step.description?.slice(0, 100)}
                  </div>
                ))}

                <h4 style={{ color: 'var(--accent-primary)', margin: '0.75rem 0 0.5rem' }}>Key Spices</h4>
                <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                  {recipe.key_spices?.map((s, i) => (
                    <span key={i} className="badge badge-saffron">{s}</span>
                  ))}
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

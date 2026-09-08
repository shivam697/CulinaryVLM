import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../api';
import { CATEGORY_EMOJIS } from '../constants/biryaniStyles';
import PageHeader from '../components/PageHeader';
import StatCard from '../components/StatCard';
import LoadingSpinner from '../components/LoadingSpinner';
import SafePrism from '../components/SafePrism';

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
    return <LoadingSpinner message="Loading recipes..." />;
  }

  return (
    <>
      {/* ── Hero with Prism background ── */}
      <div className="hero-section">
        {/* Layer 0: Prism canvas */}
        <SafePrism
          transparent={true}
          animationType="rotate"
          timeScale={0.4}
          glow={0.8}
          bloom={0.9}
          noise={0.25}
          hueShift={0}
          colorFrequency={1}
          suspendWhenOffscreen={true}
        />

        {/* Layer 1: legibility scrim */}
        <div className="hero-scrim" />

        {/* Layer 2: text content */}
        <div className="hero-content">
          <h1>Explore Biryani Intelligence</h1>
          <p>
            Discover 20 regional styles of chicken biryani — search cooking
            techniques, compare recipes, and ask reasoning questions powered
            by multimodal AI.
          </p>
          <div className="hero-cta">
            <Link to="/search" className="btn btn-primary">
              🔍 Search Techniques
            </Link>
            <Link to="/agent" className="btn btn-secondary">
              🤖 Ask the Agent
            </Link>
          </div>
        </div>
      </div>

      {/* ── Existing page content (untouched) ── */}
      <div className="page-container">
        <PageHeader
          title="Explore Biryani Styles"
          subtitle="Discover 20 regional styles of chicken biryani across India and beyond"
        />

        {/* Stats */}
        {stats && (
          <div className="grid grid-4 stagger" style={{ marginBottom: 'var(--space-8)' }}>
            <StatCard value={stats.total_videos} label="Videos Analyzed" index={0} />
            <StatCard value={Object.keys(stats.categories || {}).length} label="Regional Styles" index={1} />
            <StatCard value={Object.keys(stats.languages || {}).length} label="Languages" index={2} />
            <StatCard value={recipes.length} label="Canonical Recipes" index={3} />
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
              <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)', marginBottom: 'var(--space-3)' }}>
                <span style={{ fontSize: 'var(--text-2xl)' }}>{CATEGORY_EMOJIS[recipe.category] || '🍚'}</span>
                <div>
                  <h3 style={{ fontSize: 'var(--text-lg)' }}>{recipe.category}</h3>
                  <span style={{ fontSize: 'var(--text-sm)', color: 'var(--text-muted)' }}>{recipe.region}</span>
                </div>
              </div>

              <div style={{ display: 'flex', gap: 'var(--space-2)', flexWrap: 'wrap', marginBottom: 'var(--space-3)' }}>
                <span className="badge badge-saffron">{recipe.steps?.length || 0} steps</span>
                <span className="badge badge-blue">{recipe.rice_type?.split(' ')[0]}</span>
                {recipe.dum_method && <span className="badge badge-purple">{recipe.dum_method.split(' ')[0]} dum</span>}
              </div>

              {recipe.distinguishing_features?.slice(0, 2).map((f, i) => (
                <p key={i} style={{ fontSize: 'var(--text-sm)', color: 'var(--text-secondary)', marginBottom: 'var(--space-1)' }}>
                  • {f.length > 80 ? f.slice(0, 80) + '…' : f}
                </p>
              ))}

              {/* Expanded Detail */}
              {selectedRecipe?.category === recipe.category && (
                <div style={{ marginTop: 'var(--space-4)', paddingTop: 'var(--space-4)', borderTop: '1px solid var(--border-subtle)' }}>
                  <h4 style={{ color: 'var(--accent-primary)', marginBottom: 'var(--space-3)' }}>Key Steps</h4>
                  {recipe.steps?.filter(s => s.is_defining_step).map((step) => (
                    <div key={step.step_number} style={{ marginBottom: 'var(--space-2)', fontSize: 'var(--text-sm)' }}>
                      <span style={{ color: 'var(--accent-primary)', fontWeight: 600 }}>{step.step_number}.</span>{' '}
                      <strong>{step.action}</strong> — {step.description?.slice(0, 100)}
                    </div>
                  ))}

                  <h4 style={{ color: 'var(--accent-primary)', margin: 'var(--space-3) 0 var(--space-2)' }}>Key Spices</h4>
                  <div style={{ display: 'flex', gap: 'var(--space-2)', flexWrap: 'wrap' }}>
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
    </>
  );
}

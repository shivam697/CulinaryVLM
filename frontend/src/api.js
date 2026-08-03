const API_BASE = import.meta.env.VITE_API_URL || '';

async function request(path, options = {}) {
  const url = `${API_BASE}${path}`;
  const res = await fetch(url, {
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `API error: ${res.status}`);
  }
  return res.json();
}

export const api = {
  // Health
  health: () => request('/health'),

  // Videos
  getVideos: (params = {}) => {
    const qs = new URLSearchParams(params).toString();
    return request(`/api/v1/videos${qs ? '?' + qs : ''}`);
  },
  getVideo: (id) => request(`/api/v1/videos/${id}`),
  getVideoStats: () => request('/api/v1/videos/stats/overview'),

  // Recipes
  getRecipes: () => request('/api/v1/recipes'),
  getRecipe: (category) => request(`/api/v1/recipes/${category}`),

  // Search
  search: (query, category, topK = 10) =>
    request('/api/v1/search', {
      method: 'POST',
      body: JSON.stringify({ query, category, top_k: topK }),
    }),

  // QA
  askQuestion: (question, category) =>
    request('/api/v1/qa', {
      method: 'POST',
      body: JSON.stringify({ question, category }),
    }),

  // Compare
  compare: (categoryA, categoryB, aspect) =>
    request('/api/v1/compare', {
      method: 'POST',
      body: JSON.stringify({ category_a: categoryA, category_b: categoryB, aspect }),
    }),

  // Agent
  agentQuery: (message, sessionId) =>
    request('/api/v1/agent/query', {
      method: 'POST',
      body: JSON.stringify({ message, session_id: sessionId }),
    }),
};

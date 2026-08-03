import { useState } from 'react';
import { api } from '../api';

export default function QAPage() {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleAsk = async (e) => {
    e.preventDefault();
    if (!question.trim()) return;
    setLoading(true);
    try {
      const data = await api.askQuestion(question.trim());
      setAnswer(data);
    } catch (err) {
      setAnswer({ question: question, answer: `Error: ${err.message}`, confidence: 0 });
    } finally {
      setLoading(false);
    }
  };

  const sampleQuestions = [
    'What spices are used in Hyderabadi biryani?',
    'How is Kolkata biryani different from Hyderabadi?',
    'What is the dum cooking method in Lucknowi biryani?',
    'What makes Malabar biryani unique?',
    'How do you prepare rice for Sindhi biryani?',
  ];

  return (
    <div className="page-container">
      <div className="page-header animate-in">
        <h1>Ask About Biryani</h1>
        <p>Get answers about cooking techniques, ingredients, and regional variations</p>
      </div>

      <form onSubmit={handleAsk} style={{ marginBottom: '2rem', maxWidth: '700px' }}>
        <textarea
          className="input"
          placeholder="Ask anything about biryani... e.g., What spices make Hyderabadi biryani special?"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          rows={3}
          style={{ resize: 'vertical', marginBottom: '0.75rem' }}
        />
        <button type="submit" className="btn btn-primary" disabled={loading}>
          {loading ? '⟳ Thinking...' : '💬 Ask Question'}
        </button>
      </form>

      {answer && (
        <div className="card animate-in" style={{ maxWidth: '700px' }}>
          <div style={{ marginBottom: '1rem' }}>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Question:</p>
            <p style={{ fontWeight: 600 }}>{answer.question}</p>
          </div>

          <div style={{
            padding: '1.25rem',
            background: 'rgba(245, 158, 11, 0.05)',
            borderRadius: 'var(--radius-md)',
            borderLeft: '3px solid var(--accent-primary)',
          }}>
            <p style={{ whiteSpace: 'pre-wrap', lineHeight: 1.7 }}>{answer.answer}</p>
          </div>

          <div style={{ display: 'flex', gap: '1rem', marginTop: '1rem', alignItems: 'center' }}>
            {answer.confidence > 0 && (
              <span className="badge badge-green">
                Confidence: {Math.round(answer.confidence * 100)}%
              </span>
            )}
            {answer.model_used && (
              <span className="badge badge-blue">{answer.model_used}</span>
            )}
            {answer.sources?.map((s, i) => (
              <span key={i} className="badge badge-purple">{s.category || s.type}</span>
            ))}
          </div>
        </div>
      )}

      {!answer && !loading && (
        <div style={{ maxWidth: '700px' }}>
          <h3 style={{ marginBottom: '1rem', color: 'var(--text-secondary)', fontSize: '1rem' }}>
            💡 Try these questions:
          </h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            {sampleQuestions.map((q) => (
              <button
                key={q}
                className="btn btn-secondary"
                style={{ justifyContent: 'flex-start', textAlign: 'left' }}
                onClick={() => setQuestion(q)}
              >
                {q}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

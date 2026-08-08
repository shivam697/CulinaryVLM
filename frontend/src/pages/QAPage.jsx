import { useState } from 'react';
import { api } from '../api';
import PageHeader from '../components/PageHeader';
import SuggestionChips from '../components/SuggestionChips';

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
      <PageHeader
        title="Ask About Biryani"
        subtitle="Get answers about cooking techniques, ingredients, and regional variations"
      />

      <form onSubmit={handleAsk} style={{ marginBottom: 'var(--space-8)', maxWidth: 'var(--container-narrow)' }}>
        <textarea
          className="input"
          placeholder="Ask anything about biryani... e.g., What spices make Hyderabadi biryani special?"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          rows={3}
          style={{ resize: 'vertical', marginBottom: 'var(--space-3)' }}
        />
        <button type="submit" className="btn btn-primary" disabled={loading}>
          {loading ? '⟳ Thinking...' : '💬 Ask Question'}
        </button>
      </form>

      {answer && (
        <div className="card animate-in" style={{ maxWidth: 'var(--container-narrow)' }}>
          <div style={{ marginBottom: 'var(--space-4)' }}>
            <p style={{ fontSize: 'var(--text-sm)', color: 'var(--text-muted)' }}>Question:</p>
            <p style={{ fontWeight: 600 }}>{answer.question}</p>
          </div>

          <div style={{
            padding: 'var(--space-5)',
            background: 'rgba(245, 158, 11, 0.05)',
            borderRadius: 'var(--radius-md)',
            borderLeft: '3px solid var(--accent-primary)',
          }}>
            <p style={{ whiteSpace: 'pre-wrap', lineHeight: 1.7 }}>{answer.answer}</p>
          </div>

          <div style={{ display: 'flex', gap: 'var(--space-4)', marginTop: 'var(--space-4)', alignItems: 'center' }}>
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
        <div style={{ maxWidth: 'var(--container-narrow)' }}>
          <h3 style={{ marginBottom: 'var(--space-4)', color: 'var(--text-secondary)', fontSize: 'var(--text-md)' }}>
            💡 Try these questions:
          </h3>
          <SuggestionChips
            suggestions={sampleQuestions}
            onSelect={(q) => setQuestion(q)}
          />
        </div>
      )}
    </div>
  );
}

export default function StatCard({ value, label }) {
  return (
    <div className="stat-card animate-slide-up">
      <span className="stat-value">{value}</span>
      <span className="stat-label">{label}</span>
    </div>
  );
}

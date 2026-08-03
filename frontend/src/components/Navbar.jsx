import { NavLink } from 'react-router-dom';

export default function Navbar() {
  return (
    <nav className="navbar">
      <div className="navbar-inner">
        <NavLink to="/" className="navbar-brand">
          <span className="brand-icon">🍚</span>
          <span>Culinary<span className="brand-accent">VLM</span></span>
        </NavLink>
        <div className="navbar-links">
          <NavLink to="/" end className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
            Explore
          </NavLink>
          <NavLink to="/search" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
            Search
          </NavLink>
          <NavLink to="/qa" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
            Ask
          </NavLink>
          <NavLink to="/compare" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
            Compare
          </NavLink>
          <NavLink to="/agent" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
            🤖 Agent
          </NavLink>
        </div>
      </div>
    </nav>
  );
}

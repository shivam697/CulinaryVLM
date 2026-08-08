import { useState } from 'react';
import { NavLink } from 'react-router-dom';

export default function Navbar() {
  const [menuOpen, setMenuOpen] = useState(false);

  // Close menu on navigation
  const handleNavClick = () => setMenuOpen(false);

  return (
    <nav className="navbar" aria-label="Main navigation">
      <div className="navbar-inner">
        <NavLink to="/" className="navbar-brand" onClick={handleNavClick}>
          <span className="brand-icon">🍚</span>
          <span>Culinary<span className="brand-accent">VLM</span></span>
        </NavLink>

        <button
          className="navbar-toggle"
          onClick={() => setMenuOpen(!menuOpen)}
          aria-expanded={menuOpen}
          aria-controls="navbar-menu"
          aria-label="Toggle navigation menu"
        >
          {menuOpen ? '✕' : '☰'}
        </button>

        <div
          id="navbar-menu"
          className={`navbar-links${menuOpen ? ' open' : ''}`}
          role="menubar"
        >
          <NavLink to="/" end className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`} onClick={handleNavClick}>
            Explore
          </NavLink>
          <NavLink to="/search" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`} onClick={handleNavClick}>
            Search
          </NavLink>
          <NavLink to="/qa" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`} onClick={handleNavClick}>
            Ask
          </NavLink>
          <NavLink to="/compare" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`} onClick={handleNavClick}>
            Compare
          </NavLink>
          <NavLink to="/agent" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`} onClick={handleNavClick}>
            🤖 Agent
          </NavLink>
        </div>
      </div>
    </nav>
  );
}

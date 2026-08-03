import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import ExplorePage from './pages/ExplorePage';
import SearchPage from './pages/SearchPage';
import QAPage from './pages/QAPage';
import ComparePage from './pages/ComparePage';
import AgentPage from './pages/AgentPage';

export default function App() {
  return (
    <BrowserRouter>
      <div className="app-layout">
        <Navbar />
        <Routes>
          <Route path="/" element={<ExplorePage />} />
          <Route path="/search" element={<SearchPage />} />
          <Route path="/qa" element={<QAPage />} />
          <Route path="/compare" element={<ComparePage />} />
          <Route path="/agent" element={<AgentPage />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}

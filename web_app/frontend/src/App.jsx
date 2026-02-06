import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import { Share2, Activity } from 'lucide-react';

import HistoricalView from './pages/HistoricalView';
import PredictionView from './pages/PredictionView';

function NavBar() {
  const location = useLocation();
  return (
    <nav className="glass-panel" style={{ borderRadius: 0, borderTop: 0, borderLeft: 0, borderRight: 0 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
        <Activity className="text-blue-500" size={28} style={{ color: 'var(--accent-primary)' }} />
        <h1 style={{ margin: 0, fontSize: '1.5rem' }}>
          Flood<span className="gradient-text">Vis</span>
        </h1>
      </div>
      <div className="nav-links">
        <Link to="/" className={location.pathname === '/' ? 'active' : ''}>Historical Data</Link>
        <Link to="/predict" className={location.pathname === '/predict' ? 'active' : ''}>Live Prediction</Link>
      </div>
    </nav>
  );
}

function App() {
  return (
    <Router>
      <div className="app-container">
        <NavBar />
        <Routes>
          <Route path="/" element={<HistoricalView />} />
          <Route path="/predict" element={<PredictionView />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;

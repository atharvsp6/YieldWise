import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import './App.css';

// Import components
import Navbar from './components/Navbar';
import Home from './pages/Home';
import YieldPredictor from './pages/YieldPredictor';
import Dashboard from './pages/Dashboard';
import FinancialCalculator from './pages/FinancialCalculator';
import DiseaseDetection from './pages/DiseaseDetection';

function App() {
  return (
    <Router>
      <div className="App">
        <Navbar />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/home" element={<Home />} />
            <Route path="/yield-predictor" element={<YieldPredictor />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/financial-calculator" element={<FinancialCalculator />} />
            <Route path="/disease-detection" element={<DiseaseDetection />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;

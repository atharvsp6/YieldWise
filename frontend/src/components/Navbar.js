import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import ThemeToggle from './ThemeToggle';
import LanguageSwitcher from './LanguageSwitcher';
import './Navbar.css';

const Navbar = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const location = useLocation();
  const { t } = useTranslation();

  const toggleMenu = () => {
    setIsMenuOpen(!isMenuOpen);
  };

  const isActive = (path) => {
    return location.pathname === path;
  };

  return (
    <nav className="navbar">
      <div className="navbar-container">
        <div className="navbar-brand">
          <Link to="/" className="brand-link">
            <span className="brand-icon">🌾</span>
            <span className="brand-text">YieldWise</span>
          </Link>
        </div>
        
        <div className={`navbar-menu ${isMenuOpen ? 'active' : ''}`}>
          <ul className="navbar-nav">
            <li className="nav-item">
              <Link 
                to="/home" 
                className={`nav-link ${isActive('/home') || isActive('/') ? 'active' : ''}`}
                onClick={() => setIsMenuOpen(false)}
              >
                <span className="nav-icon">🏠</span>
                {t('nav.home')}
              </Link>
            </li>
            <li className="nav-item">
              <Link 
                to="/yield-predictor" 
                className={`nav-link ${isActive('/yield-predictor') ? 'active' : ''}`}
                onClick={() => setIsMenuOpen(false)}
              >
                <span className="nav-icon">📊</span>
                {t('nav.yieldPredictor')}
              </Link>
            </li>
            <li className="nav-item">
              <Link 
                to="/dashboard" 
                className={`nav-link ${isActive('/dashboard') ? 'active' : ''}`}
                onClick={() => setIsMenuOpen(false)}
              >
                <span className="nav-icon">📈</span>
                {t('nav.dashboard')}
              </Link>
            </li>
            <li className="nav-item">
              <Link 
                to="/financial-calculator" 
                className={`nav-link ${isActive('/financial-calculator') ? 'active' : ''}`}
                onClick={() => setIsMenuOpen(false)}
              >
                <span className="nav-icon">💰</span>
                {t('nav.financialCalculator')}
              </Link>
            </li>
            <li className="nav-item">
              <Link 
                to="/disease-detection" 
                className={`nav-link ${isActive('/disease-detection') ? 'active' : ''}`}
                onClick={() => setIsMenuOpen(false)}
              >
                <span className="nav-icon">🔬</span>
                {t('nav.diseaseDetection')}
              </Link>
            </li>
          </ul>
        </div>
        
        <div className="navbar-controls">
          <LanguageSwitcher />
          <ThemeToggle />
          <div className="navbar-toggle" onClick={toggleMenu}>
            <span className={`toggle-bar ${isMenuOpen ? 'active' : ''}`}></span>
            <span className={`toggle-bar ${isMenuOpen ? 'active' : ''}`}></span>
            <span className={`toggle-bar ${isMenuOpen ? 'active' : ''}`}></span>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;

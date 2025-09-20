import React from 'react';
import { Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import '../styles/Home.css';

const Home = () => {
  const { t } = useTranslation();

  return (
    <div className="home">
      {/* Hero Section */}
      <section className="hero-section">
        <div className="hero-content">
          <h1 className="hero-title">
            <span className="hero-icon">🌾</span>
            {t('home.title')}
          </h1>
          <p className="hero-subtitle">
            {t('home.subtitle')}
          </p>
          <p className="hero-description">
            {t('home.description')}
          </p>
          <div className="hero-actions">
            <Link to="/yield-predictor" className="btn btn-primary">
              🎯 {t('home.getStarted')}
            </Link>
            <Link to="/dashboard" className="btn btn-secondary">
              📊 {t('home.learnMore')}
            </Link>
          </div>
        </div>
        <div className="hero-image">
          <div className="hero-visual">🌱🚜🌾</div>
        </div>
      </section>

      {/* Features Section */}
      <section className="features-section">
        <div className="container">
          <h2 className="section-title">Powerful Features for Smart Farming</h2>
          <div className="features-grid">
            <div className="feature-card">
              <div className="feature-icon">🎯</div>
              <h3>{t('home.features.yieldPrediction.title')}</h3>
              <p>{t('home.features.yieldPrediction.description')}</p>
              <Link to="/yield-predictor" className="feature-link">Try Now →</Link>
            </div>
            
            <div className="feature-card">
              <div className="feature-icon">💰</div>
              <h3>{t('home.features.financialAnalysis.title')}</h3>
              <p>{t('home.features.financialAnalysis.description')}</p>
              <Link to="/financial-calculator" className="feature-link">Calculate →</Link>
            </div>
            
            <div className="feature-card">
              <div className="feature-icon">🔬</div>
              <h3>{t('home.features.diseaseDetection.title')}</h3>
              <p>{t('home.features.diseaseDetection.description')}</p>
              <Link to="/disease-detection" className="feature-link">Detect →</Link>
            </div>
            
            <div className="feature-card">
              <div className="feature-icon">📈</div>
              <h3>Analytics Dashboard</h3>
              <p>Comprehensive insights and analytics for all your agricultural data</p>
              <Link to="/dashboard" className="feature-link">Explore →</Link>
            </div>
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section className="how-it-works-section">
        <div className="container">
          <h2 className="section-title">How YieldWise Works</h2>
          <div className="steps-grid">
            <div className="step-item">
              <div className="step-number">1</div>
              <div className="step-content">
                <h3>Input Your Data</h3>
                <p>Enter crop details, location, soil parameters, and farming practices</p>
              </div>
            </div>
            
            <div className="step-item">
              <div className="step-number">2</div>
              <div className="step-content">
                <h3>AI Analysis</h3>
                <p>Our ML models analyze weather patterns, soil conditions, and historical data</p>
              </div>
            </div>
            
            <div className="step-item">
              <div className="step-number">3</div>
              <div className="step-content">
                <h3>Get Predictions</h3>
                <p>Receive accurate yield predictions with confidence intervals</p>
              </div>
            </div>
            
            <div className="step-item">
              <div className="step-number">4</div>
              <div className="step-content">
                <h3>Smart Recommendations</h3>
                <p>Get AI-powered recommendations to optimize your farming practices</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Getting Started Section */}
      <section className="getting-started-section">
        <div className="container">
          <h2 className="section-title">Getting Started</h2>
          <div className="getting-started-content">
            <div className="instructions">
              <div className="instruction-item">
                <h3>🌱 For New Users</h3>
                <ul>
                  <li>Start with our <Link to="/yield-predictor">Yield Predictor</Link> to get familiar</li>
                  <li>Enter your crop and location details</li>
                  <li>Review the predictions and recommendations</li>
                  <li>Explore other features like financial planning and disease detection</li>
                </ul>
              </div>
              
              <div className="instruction-item">
                <h3>📊 Advanced Features</h3>
                <ul>
                  <li>Use the <Link to="/dashboard">Dashboard</Link> for comprehensive analytics</li>
                  <li>Plan your finances with the <Link to="/financial-calculator">Financial Calculator</Link></li>
                  <li>Monitor plant health with <Link to="/disease-detection">Disease Detection</Link></li>
                  <li>Track your progress over multiple seasons</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Home;
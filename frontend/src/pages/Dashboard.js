import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import '../styles/Dashboard.css';

const Dashboard = () => {
  const [stats, setStats] = useState({
    totalPredictions: 0,
    averageYield: 0,
    totalArea: 0,
    mostPopularCrop: 'N/A'
  });

  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Simulate loading dashboard data
    const timer = setTimeout(() => {
      setStats({
        totalPredictions: 42,
        averageYield: 3.8,
        totalArea: 125.5,
        mostPopularCrop: 'Wheat'
      });
      setLoading(false);
    }, 1000);

    return () => clearTimeout(timer);
  }, []);

  const quickActions = [
    {
      title: 'New Prediction',
      description: 'Start a new crop yield prediction',
      icon: '🎯',
      link: '/yield-predictor',
      color: 'blue'
    },
    {
      title: 'Financial Analysis',
      description: 'Calculate costs and profits',
      icon: '💰',
      link: '/financial-calculator',
      color: 'green'
    },
    {
      title: 'Disease Detection',
      description: 'Analyze plant health',
      icon: '🔬',
      link: '/disease-detection',
      color: 'red'
    }
  ];

  const recentInsights = [
    {
      type: 'prediction',
      title: 'High Yield Expected',
      description: 'Your wheat prediction shows 15% above average yield',
      time: '2 hours ago',
      icon: '📈'
    },
    {
      type: 'weather',
      title: 'Rainfall Alert',
      description: 'Heavy rainfall expected in your region next week',
      time: '5 hours ago',
      icon: '🌧️'
    },
    {
      type: 'market',
      title: 'Price Update',
      description: 'Cotton prices increased by 8% in local markets',
      time: '1 day ago',
      icon: '💹'
    }
  ];

  const cropData = [
    { crop: 'Wheat', area: 45.2, yield: 4.2, status: 'Excellent' },
    { crop: 'Rice', area: 32.8, yield: 3.8, status: 'Good' },
    { crop: 'Maize', area: 28.1, yield: 3.5, status: 'Average' },
    { crop: 'Cotton', area: 19.4, yield: 2.9, status: 'Good' }
  ];

  if (loading) {
    return (
      <div className="dashboard loading">
        <div className="loading-spinner">
          <div className="spinner"></div>
          <p>Loading dashboard...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="dashboard">
      <div className="container">
        {/* Header */}
        <header className="dashboard-header">
          <div className="header-content">
            <h1>📊 Agricultural Dashboard</h1>
            <p>Your comprehensive farming intelligence center</p>
          </div>
          <div className="header-stats">
            <div className="stat-chip">
              <span className="stat-label">Last Updated</span>
              <span className="stat-value">Just now</span>
            </div>
          </div>
        </header>

        {/* Stats Grid */}
        <section className="stats-section">
          <div className="stats-grid">
            <div className="stat-card">
              <div className="stat-icon">🎯</div>
              <div className="stat-content">
                <h3>Total Predictions</h3>
                <div className="stat-value">{stats.totalPredictions}</div>
                <div className="stat-change positive">+12% this month</div>
              </div>
            </div>

            <div className="stat-card">
              <div className="stat-icon">🌾</div>
              <div className="stat-content">
                <h3>Average Yield</h3>
                <div className="stat-value">{stats.averageYield} t/ha</div>
                <div className="stat-change positive">+8% vs last season</div>
              </div>
            </div>

            <div className="stat-card">
              <div className="stat-icon">📏</div>
              <div className="stat-content">
                <h3>Total Area</h3>
                <div className="stat-value">{stats.totalArea} ha</div>
                <div className="stat-change neutral">No change</div>
              </div>
            </div>

            <div className="stat-card">
              <div className="stat-icon">🏆</div>
              <div className="stat-content">
                <h3>Top Crop</h3>
                <div className="stat-value">{stats.mostPopularCrop}</div>
                <div className="stat-change positive">Highest performer</div>
              </div>
            </div>
          </div>
        </section>

        {/* Quick Actions */}
        <section className="quick-actions-section">
          <h2>⚡ Quick Actions</h2>
          <div className="quick-actions-grid">
            {quickActions.map((action, index) => (
              <Link
                key={index}
                to={action.link}
                className={`action-card ${action.color}`}
              >
                <div className="action-icon">{action.icon}</div>
                <div className="action-content">
                  <h3>{action.title}</h3>
                  <p>{action.description}</p>
                </div>
                <div className="action-arrow">→</div>
              </Link>
            ))}
          </div>
        </section>

        {/* Main Content Grid */}
        <div className="main-content-grid">
          {/* Crop Overview */}
          <section className="crop-overview-section">
            <div className="section-card">
              <h2>🌱 Crop Overview</h2>
              <div className="crop-table">
                <div className="table-header">
                  <div>Crop</div>
                  <div>Area (ha)</div>
                  <div>Yield (t/ha)</div>
                  <div>Status</div>
                </div>
                {cropData.map((crop, index) => (
                  <div key={index} className="table-row">
                    <div className="crop-name">{crop.crop}</div>
                    <div>{crop.area}</div>
                    <div>{crop.yield}</div>
                    <div className={`status ${crop.status.toLowerCase()}`}>
                      {crop.status}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </section>

          {/* Recent Insights */}
          <section className="insights-section">
            <div className="section-card">
              <h2>💡 Recent Insights</h2>
              <div className="insights-list">
                {recentInsights.map((insight, index) => (
                  <div key={index} className={`insight-item ${insight.type}`}>
                    <div className="insight-icon">{insight.icon}</div>
                    <div className="insight-content">
                      <h4>{insight.title}</h4>
                      <p>{insight.description}</p>
                      <span className="insight-time">{insight.time}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </section>
        </div>

        {/* Weather Widget */}
        <section className="weather-section">
          <div className="section-card">
            <h2>🌤️ Weather Overview</h2>
            <div className="weather-grid">
              <div className="weather-item">
                <div className="weather-icon">☀️</div>
                <div className="weather-info">
                  <span className="weather-label">Temperature</span>
                  <span className="weather-value">28°C</span>
                </div>
              </div>
              <div className="weather-item">
                <div className="weather-icon">💧</div>
                <div className="weather-info">
                  <span className="weather-label">Humidity</span>
                  <span className="weather-value">65%</span>
                </div>
              </div>
              <div className="weather-item">
                <div className="weather-icon">🌬️</div>
                <div className="weather-info">
                  <span className="weather-label">Wind Speed</span>
                  <span className="weather-value">12 km/h</span>
                </div>
              </div>
              <div className="weather-item">
                <div className="weather-icon">🌧️</div>
                <div className="weather-info">
                  <span className="weather-label">Precipitation</span>
                  <span className="weather-value">0 mm</span>
                </div>
              </div>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
};

export default Dashboard;
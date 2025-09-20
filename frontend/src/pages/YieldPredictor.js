import React, { useState } from 'react';
import axios from 'axios';
import { useTranslation } from 'react-i18next';
import '../styles/YieldPredictor.css';

const YieldPredictor = () => {
  const { t } = useTranslation();
  const [formData, setFormData] = useState({
    crop: '',
    season: '',
    state: '',
    area: '',
    rainfall: '',
    fertilizer: '',
    pesticide: ''
  });
  
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setPrediction(null);

    try {
      const response = await axios.post('/api/predict_yield', {
        crop: formData.crop,
        season: formData.season,
        state: formData.state,
        area: parseFloat(formData.area),
        rainfall: parseFloat(formData.rainfall),
        fertilizer: parseFloat(formData.fertilizer),
        pesticide: parseFloat(formData.pesticide)
      });

      if (response.data.status === 'success') {
        setPrediction(response.data);
      } else {
        setError(response.data.message || t('yieldPredictor.results.error'));
      }
    } catch (err) {
      setError(t('yieldPredictor.results.error'));
      console.error('Prediction error:', err);
    } finally {
      setLoading(false);
    }
  };

  const crops = [
    'Rice', 'Wheat', 'Maize', 'Cotton', 'Sugarcane', 
    'Groundnut', 'Bajra', 'Barley', 'Jowar', 'Ragi'
  ];

  const seasons = [
    'Kharif', 'Rabi', 'Zaid', 'Summer', 'Winter'
  ];

  const states = [
    'Andhra Pradesh', 'Bihar', 'Gujarat', 'Haryana', 'Karnataka',
    'Madhya Pradesh', 'Maharashtra', 'Odisha', 'Punjab', 'Rajasthan',
    'Tamil Nadu', 'Uttar Pradesh', 'West Bengal', 'Telangana'
  ];

  return (
    <div className="yield-predictor">
      <div className="container">
        {/* Header */}
        <header className="page-header">
          <h1>🌾 Crop Yield Predictor</h1>
          <p>Get AI-powered predictions for your crop yields based on multiple factors</p>
        </header>

        <div className="predictor-content">
          {/* Prediction Form */}
          <div className="form-section">
            <div className="form-card">
              <h2>📋 Enter Crop Details</h2>
              <form onSubmit={handleSubmit} className="prediction-form">
                <div className="form-grid">
                  <div className="form-group">
                    <label htmlFor="crop">
                      <span className="label-icon">🌱</span>
                      Crop Type
                    </label>
                    <select
                      id="crop"
                      name="crop"
                      value={formData.crop}
                      onChange={handleChange}
                      required
                    >
                      <option value="">Select Crop</option>
                      {crops.map(crop => (
                        <option key={crop} value={crop}>{crop}</option>
                      ))}
                    </select>
                  </div>

                  <div className="form-group">
                    <label htmlFor="season">
                      <span className="label-icon">🗓️</span>
                      Season
                    </label>
                    <select
                      id="season"
                      name="season"
                      value={formData.season}
                      onChange={handleChange}
                      required
                    >
                      <option value="">Select Season</option>
                      {seasons.map(season => (
                        <option key={season} value={season}>{season}</option>
                      ))}
                    </select>
                  </div>

                  <div className="form-group">
                    <label htmlFor="state">
                      <span className="label-icon">📍</span>
                      State
                    </label>
                    <select
                      id="state"
                      name="state"
                      value={formData.state}
                      onChange={handleChange}
                      required
                    >
                      <option value="">Select State</option>
                      {states.map(state => (
                        <option key={state} value={state}>{state}</option>
                      ))}
                    </select>
                  </div>

                  <div className="form-group">
                    <label htmlFor="area">
                      <span className="label-icon">📏</span>
                      Area (hectares)
                    </label>
                    <input
                      type="number"
                      id="area"
                      name="area"
                      value={formData.area}
                      onChange={handleChange}
                      placeholder="e.g., 5.5"
                      min="0.1"
                      step="0.1"
                      required
                    />
                  </div>

                  <div className="form-group">
                    <label htmlFor="rainfall">
                      <span className="label-icon">🌧️</span>
                      Rainfall (mm)
                    </label>
                    <input
                      type="number"
                      id="rainfall"
                      name="rainfall"
                      value={formData.rainfall}
                      onChange={handleChange}
                      placeholder="e.g., 800"
                      min="0"
                      required
                    />
                  </div>

                  <div className="form-group">
                    <label htmlFor="fertilizer">
                      <span className="label-icon">🧪</span>
                      Fertilizer (kg/ha)
                    </label>
                    <input
                      type="number"
                      id="fertilizer"
                      name="fertilizer"
                      value={formData.fertilizer}
                      onChange={handleChange}
                      placeholder="e.g., 150"
                      min="0"
                      step="0.1"
                      required
                    />
                  </div>

                  <div className="form-group">
                    <label htmlFor="pesticide">
                      <span className="label-icon">🐛</span>
                      Pesticide (kg/ha)
                    </label>
                    <input
                      type="number"
                      id="pesticide"
                      name="pesticide"
                      value={formData.pesticide}
                      onChange={handleChange}
                      placeholder="e.g., 2.5"
                      min="0"
                      step="0.1"
                      required
                    />
                  </div>
                </div>

                <button 
                  type="submit" 
                  className="predict-btn"
                  disabled={loading}
                >
                  {loading ? (
                    <>
                      <span className="spinner"></span>
                      Analyzing...
                    </>
                  ) : (
                    <>
                      🔮 Predict Yield
                    </>
                  )}
                </button>
              </form>
            </div>
          </div>

          {/* Results Section */}
          <div className="results-section">
            {error && (
              <div className="error-card">
                <h3>❌ Error</h3>
                <p>{error}</p>
              </div>
            )}

            {prediction && (
              <div className="results-card">
                <h2>🎯 Prediction Results</h2>
                
                <div className="prediction-summary">
                  <div className="yield-display">
                    <div className="yield-value">
                      {prediction.predicted_yield ? prediction.predicted_yield.toFixed(2) : '0.00'}
                    </div>
                    <div className="yield-unit">tons/hectare</div>
                  </div>
                  
                  <div className="prediction-details">
                    <div className="detail-item">
                      <span className="detail-label">Confidence Range:</span>
                      <span className="detail-value">
                        {prediction.confidence_interval && Array.isArray(prediction.confidence_interval) ? 
                          `${prediction.confidence_interval[0]?.toFixed(2)} - ${prediction.confidence_interval[1]?.toFixed(2)}` 
                          : 'N/A'
                        }
                      </span>
                    </div>
                    <div className="detail-item">
                      <span className="detail-label">Total Production:</span>
                      <span className="detail-value">
                        {(prediction.predicted_yield && formData.area) ? 
                          (prediction.predicted_yield * parseFloat(formData.area)).toFixed(2) + ' tons'
                          : 'N/A'
                        }
                      </span>
                    </div>
                    <div className="detail-item">
                      <span className="detail-label">Status:</span>
                      <span className={`detail-value ${(prediction.prediction_status || prediction.status || '').toLowerCase()}`}>
                        {prediction.prediction_status || prediction.status || 'Unknown'}
                      </span>
                    </div>
                  </div>
                </div>

                {prediction.recommendations && (
                  <div className="recommendations">
                    <h3>🤖 AI Recommendations</h3>
                    <div className="recommendations-content">
                      {Object.entries(prediction.recommendations).map(([category, recommendations]) => (
                        <div key={category} className="recommendation-category">
                          <h4>{category.replace(/_/g, ' ')}</h4>
                          {typeof recommendations === 'object' && recommendations !== null ? (
                            <ul>
                              {Object.entries(recommendations).map(([key, value]) => (
                                <li key={key}>
                                  <strong>{key.replace(/_/g, ' ')}:</strong> {value}
                                </li>
                              ))}
                            </ul>
                          ) : (
                            <p>{recommendations}</p>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {!prediction && !error && !loading && (
              <div className="placeholder-card">
                <div className="placeholder-content">
                  <div className="placeholder-icon">📊</div>
                  <h3>Ready for Prediction</h3>
                  <p>Fill out the form and click "Predict Yield" to get AI-powered crop yield predictions</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default YieldPredictor;
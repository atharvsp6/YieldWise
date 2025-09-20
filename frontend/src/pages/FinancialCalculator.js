import React, { useState } from 'react';
import axios from 'axios';
import '../styles/FinancialCalculator.css';

const FinancialCalculator = () => {
  const [formData, setFormData] = useState({
    cropName: '',
    area: '',
    state: '',
    seedCost: '',
    laborCost: '',
    fertilizerCost: '',
    pesticideCost: '',
    equipmentCost: '',
    irrigationCost: '',
    expectedYield: '',
    marketPrice: ''
  });

  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const calculateFinance = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setResults(null);

    try {
      // Calculate locally since backend endpoint might not exist
      const totalCosts = 
        parseFloat(formData.seedCost || 0) +
        parseFloat(formData.laborCost || 0) +
        parseFloat(formData.fertilizerCost || 0) +
        parseFloat(formData.pesticideCost || 0) +
        parseFloat(formData.equipmentCost || 0) +
        parseFloat(formData.irrigationCost || 0);

      const totalRevenue = parseFloat(formData.expectedYield || 0) * parseFloat(formData.marketPrice || 0);
      const profit = totalRevenue - totalCosts;
      const roi = totalCosts > 0 ? (profit / totalCosts) * 100 : 0;

      const costBreakdown = {
        seeds: parseFloat(formData.seedCost || 0),
        labor: parseFloat(formData.laborCost || 0),
        fertilizer: parseFloat(formData.fertilizerCost || 0),
        pesticide: parseFloat(formData.pesticideCost || 0),
        equipment: parseFloat(formData.equipmentCost || 0),
        irrigation: parseFloat(formData.irrigationCost || 0)
      };

      setResults({
        totalCosts,
        totalRevenue,
        profit,
        roi,
        costBreakdown,
        profitability: profit > 0 ? 'Profitable' : 'Loss'
      });
    } catch (err) {
      setError('Error calculating finance. Please check your inputs.');
      console.error('Finance calculation error:', err);
    } finally {
      setLoading(false);
    }
  };

  const formatCurrency = (amount) => {
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(amount);
  };

  return (
    <div className="financial-calculator">
      <div className="container">
        {/* Header */}
        <header className="page-header">
          <h1>💰 Financial Calculator</h1>
          <p>Calculate costs, profits, and ROI for your agricultural investments</p>
        </header>

        <div className="calculator-content">
          {/* Form Section */}
          <div className="form-section">
            <div className="form-card">
              <h2>📋 Enter Farming Details</h2>
              <form onSubmit={calculateFinance} className="finance-form">
                <div className="form-grid">
                  <div className="form-group">
                    <label htmlFor="cropName">
                      <span className="label-icon">🌱</span>
                      Crop Name
                    </label>
                    <input
                      type="text"
                      id="cropName"
                      name="cropName"
                      value={formData.cropName}
                      onChange={handleChange}
                      placeholder="e.g., Wheat"
                      required
                    />
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
                      placeholder="e.g., 5"
                      min="0.1"
                      step="0.1"
                      required
                    />
                  </div>

                  <div className="form-group">
                    <label htmlFor="state">
                      <span className="label-icon">📍</span>
                      State
                    </label>
                    <input
                      type="text"
                      id="state"
                      name="state"
                      value={formData.state}
                      onChange={handleChange}
                      placeholder="e.g., Punjab"
                      required
                    />
                  </div>

                  <div className="form-group">
                    <label htmlFor="expectedYield">
                      <span className="label-icon">🌾</span>
                      Expected Yield (tons)
                    </label>
                    <input
                      type="number"
                      id="expectedYield"
                      name="expectedYield"
                      value={formData.expectedYield}
                      onChange={handleChange}
                      placeholder="e.g., 15"
                      min="0"
                      step="0.1"
                      required
                    />
                  </div>

                  <div className="form-group">
                    <label htmlFor="marketPrice">
                      <span className="label-icon">💹</span>
                      Market Price (₹/ton)
                    </label>
                    <input
                      type="number"
                      id="marketPrice"
                      name="marketPrice"
                      value={formData.marketPrice}
                      onChange={handleChange}
                      placeholder="e.g., 25000"
                      min="0"
                      required
                    />
                  </div>
                </div>

                <h3>💸 Cost Breakdown</h3>
                <div className="form-grid">
                  <div className="form-group">
                    <label htmlFor="seedCost">
                      <span className="label-icon">🌰</span>
                      Seed Cost (₹)
                    </label>
                    <input
                      type="number"
                      id="seedCost"
                      name="seedCost"
                      value={formData.seedCost}
                      onChange={handleChange}
                      placeholder="e.g., 15000"
                      min="0"
                    />
                  </div>

                  <div className="form-group">
                    <label htmlFor="laborCost">
                      <span className="label-icon">👨‍🌾</span>
                      Labor Cost (₹)
                    </label>
                    <input
                      type="number"
                      id="laborCost"
                      name="laborCost"
                      value={formData.laborCost}
                      onChange={handleChange}
                      placeholder="e.g., 30000"
                      min="0"
                    />
                  </div>

                  <div className="form-group">
                    <label htmlFor="fertilizerCost">
                      <span className="label-icon">🧪</span>
                      Fertilizer Cost (₹)
                    </label>
                    <input
                      type="number"
                      id="fertilizerCost"
                      name="fertilizerCost"
                      value={formData.fertilizerCost}
                      onChange={handleChange}
                      placeholder="e.g., 25000"
                      min="0"
                    />
                  </div>

                  <div className="form-group">
                    <label htmlFor="pesticideCost">
                      <span className="label-icon">🐛</span>
                      Pesticide Cost (₹)
                    </label>
                    <input
                      type="number"
                      id="pesticideCost"
                      name="pesticideCost"
                      value={formData.pesticideCost}
                      onChange={handleChange}
                      placeholder="e.g., 8000"
                      min="0"
                    />
                  </div>

                  <div className="form-group">
                    <label htmlFor="equipmentCost">
                      <span className="label-icon">🚜</span>
                      Equipment Cost (₹)
                    </label>
                    <input
                      type="number"
                      id="equipmentCost"
                      name="equipmentCost"
                      value={formData.equipmentCost}
                      onChange={handleChange}
                      placeholder="e.g., 20000"
                      min="0"
                    />
                  </div>

                  <div className="form-group">
                    <label htmlFor="irrigationCost">
                      <span className="label-icon">💧</span>
                      Irrigation Cost (₹)
                    </label>
                    <input
                      type="number"
                      id="irrigationCost"
                      name="irrigationCost"
                      value={formData.irrigationCost}
                      onChange={handleChange}
                      placeholder="e.g., 12000"
                      min="0"
                    />
                  </div>
                </div>

                <button 
                  type="submit" 
                  className="calculate-btn"
                  disabled={loading}
                >
                  {loading ? (
                    <>
                      <span className="spinner"></span>
                      Calculating...
                    </>
                  ) : (
                    <>
                      🧮 Calculate Finance
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

            {results && (
              <div className="results-card">
                <h2>📊 Financial Analysis</h2>
                
                <div className="financial-summary">
                  <div className="summary-grid">
                    <div className="summary-item revenue">
                      <div className="summary-icon">💰</div>
                      <div className="summary-content">
                        <span className="summary-label">Total Revenue</span>
                        <span className="summary-value">{formatCurrency(results.totalRevenue)}</span>
                      </div>
                    </div>

                    <div className="summary-item costs">
                      <div className="summary-icon">💸</div>
                      <div className="summary-content">
                        <span className="summary-label">Total Costs</span>
                        <span className="summary-value">{formatCurrency(results.totalCosts)}</span>
                      </div>
                    </div>

                    <div className={`summary-item profit ${results.profit >= 0 ? 'positive' : 'negative'}`}>
                      <div className="summary-icon">{results.profit >= 0 ? '📈' : '📉'}</div>
                      <div className="summary-content">
                        <span className="summary-label">Net Profit</span>
                        <span className="summary-value">{formatCurrency(results.profit)}</span>
                      </div>
                    </div>

                    <div className={`summary-item roi ${results.roi >= 0 ? 'positive' : 'negative'}`}>
                      <div className="summary-icon">🎯</div>
                      <div className="summary-content">
                        <span className="summary-label">ROI</span>
                        <span className="summary-value">{results.roi.toFixed(1)}%</span>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="cost-breakdown">
                  <h3>📋 Cost Breakdown</h3>
                  <div className="breakdown-grid">
                    {Object.entries(results.costBreakdown).map(([category, amount]) => (
                      <div key={category} className="breakdown-item">
                        <span className="breakdown-category">
                          {category.charAt(0).toUpperCase() + category.slice(1)}
                        </span>
                        <span className="breakdown-amount">{formatCurrency(amount)}</span>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="profitability-status">
                  <div className={`status-badge ${results.profit >= 0 ? 'profitable' : 'loss'}`}>
                    {results.profitability}
                  </div>
                  <p className="status-description">
                    {results.profit >= 0 
                      ? 'This farming venture shows positive returns on investment.'
                      : 'This farming venture may not be profitable with current parameters.'
                    }
                  </p>
                </div>
              </div>
            )}

            {!results && !error && !loading && (
              <div className="placeholder-card">
                <div className="placeholder-content">
                  <div className="placeholder-icon">📊</div>
                  <h3>Ready for Analysis</h3>
                  <p>Fill out the form and click "Calculate Finance" to get detailed financial analysis</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default FinancialCalculator;
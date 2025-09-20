// yieldService.js - Yield prediction API service
import api from './api';

export const yieldService = {
  // Predict crop yield
  predictYield: async (yieldData) => {
    try {
      const response = await api.post('/api/predict_yield', yieldData);
      return response.data;
    } catch (error) {
      console.error('Yield prediction error:', error);
      throw error;
    }
  },

  // Get data statistics
  getDataStats: async () => {
    try {
      const response = await api.get('/api/data_stats');
      return response.data;
    } catch (error) {
      console.error('Data stats error:', error);
      throw error;
    }
  },

  // Validate yield prediction data
  validateYieldData: (data) => {
    const required = ['crop', 'season', 'state', 'area', 'rainfall', 'fertilizer', 'pesticide'];
    const missing = required.filter(field => !data[field]);
    
    if (missing.length > 0) {
      throw new Error(`Missing required fields: ${missing.join(', ')}`);
    }

    // Validate numeric fields
    const numericFields = ['area', 'rainfall', 'fertilizer', 'pesticide'];
    for (const field of numericFields) {
      const value = parseFloat(data[field]);
      if (isNaN(value) || value < 0) {
        throw new Error(`${field} must be a valid positive number`);
      }
    }

    return true;
  }
};

export default yieldService;
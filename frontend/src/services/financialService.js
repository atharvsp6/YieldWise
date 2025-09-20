// financialService.js - Financial calculation API service
import api from './api';

export const financialService = {
  // Calculate financial analysis
  calculateFinance: async (financialData) => {
    try {
      // For now, calculate locally since backend endpoint may not exist
      const result = financialService.calculateLocally(financialData);
      return result;
    } catch (error) {
      console.error('Financial calculation error:', error);
      throw error;
    }
  },

  // Local calculation method
  calculateLocally: (data) => {
    const {
      expectedYield,
      marketPrice,
      seedCost = 0,
      laborCost = 0,
      fertilizerCost = 0,
      pesticideCost = 0,
      equipmentCost = 0,
      irrigationCost = 0
    } = data;

    // Calculate costs
    const totalCosts = 
      parseFloat(seedCost) +
      parseFloat(laborCost) +
      parseFloat(fertilizerCost) +
      parseFloat(pesticideCost) +
      parseFloat(equipmentCost) +
      parseFloat(irrigationCost);

    // Calculate revenue
    const totalRevenue = parseFloat(expectedYield) * parseFloat(marketPrice);

    // Calculate profit and ROI
    const profit = totalRevenue - totalCosts;
    const roi = totalCosts > 0 ? (profit / totalCosts) * 100 : 0;

    // Cost breakdown
    const costBreakdown = {
      seeds: parseFloat(seedCost),
      labor: parseFloat(laborCost),
      fertilizer: parseFloat(fertilizerCost),
      pesticide: parseFloat(pesticideCost),
      equipment: parseFloat(equipmentCost),
      irrigation: parseFloat(irrigationCost)
    };

    return {
      status: 'success',
      totalCosts,
      totalRevenue,
      profit,
      roi,
      costBreakdown,
      profitability: profit > 0 ? 'Profitable' : 'Loss',
      recommendations: financialService.generateRecommendations(profit, roi, costBreakdown)
    };
  },

  // Generate financial recommendations
  generateRecommendations: (profit, roi, costBreakdown) => {
    const recommendations = [];

    if (profit < 0) {
      recommendations.push('Consider reducing costs or finding higher market prices');
    }

    if (roi < 20) {
      recommendations.push('ROI is below 20%. Consider optimizing resource allocation');
    }

    // Check highest cost categories
    const sortedCosts = Object.entries(costBreakdown)
      .filter(([_, cost]) => cost > 0)
      .sort(([_, a], [__, b]) => b - a);

    if (sortedCosts.length > 0) {
      const highestCost = sortedCosts[0];
      recommendations.push(`${highestCost[0]} is your highest cost category. Consider optimization strategies`);
    }

    return recommendations;
  },

  // Validate financial data
  validateFinancialData: (data) => {
    const required = ['cropName', 'area', 'expectedYield', 'marketPrice'];
    const missing = required.filter(field => !data[field]);
    
    if (missing.length > 0) {
      throw new Error(`Missing required fields: ${missing.join(', ')}`);
    }

    // Validate numeric fields
    const numericFields = ['area', 'expectedYield', 'marketPrice'];
    for (const field of numericFields) {
      const value = parseFloat(data[field]);
      if (isNaN(value) || value <= 0) {
        throw new Error(`${field} must be a valid positive number`);
      }
    }

    return true;
  },

  // Format currency for display
  formatCurrency: (amount, currency = 'INR') => {
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: currency,
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(amount);
  }
};

export default financialService;
// diseaseService.js - Disease detection API service
import api from './api';

export const diseaseService = {
  // Detect plant disease from image
  detectDisease: async (imageFile) => {
    try {
      const formData = new FormData();
      formData.append('file', imageFile);

      const response = await api.post('/api/detect_disease', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 60000, // Longer timeout for image processing
      });

      return response.data;
    } catch (error) {
      console.error('Disease detection error:', error);
      throw error;
    }
  },

  // Validate image file
  validateImageFile: (file) => {
    if (!file) {
      throw new Error('No file selected');
    }

    // Check file type
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
    if (!allowedTypes.includes(file.type)) {
      throw new Error('Please select a valid image file (JPG, PNG, or WebP)');
    }

    // Check file size (max 10MB)
    const maxSize = 10 * 1024 * 1024; // 10MB in bytes
    if (file.size > maxSize) {
      throw new Error('File size must be less than 10MB');
    }

    return true;
  },

  // Get mock disease data (for demo purposes)
  getMockDiseaseResult: () => {
    return {
      status: 'success',
      disease: 'Leaf Spot',
      confidence: 0.87,
      severity: 'Moderate',
      treatment: {
        immediate: [
          'Remove affected leaves immediately',
          'Improve air circulation around plants',
          'Avoid overhead watering'
        ],
        preventive: [
          'Apply fungicide spray every 14 days',
          'Maintain proper plant spacing',
          'Use disease-resistant varieties'
        ],
        longTerm: [
          'Crop rotation with non-host plants',
          'Improve soil drainage',
          'Regular monitoring and early detection'
        ]
      },
      description: 'Leaf spot is a common fungal disease that affects many crops. Early detection and proper treatment can prevent its spread.'
    };
  }
};

export default diseaseService;
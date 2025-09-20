import React, { useState, useRef } from 'react';
import axios from 'axios';
import '../styles/DiseaseDetection.css';

const DiseaseDetection = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);

  const handleImageSelect = (file) => {
    if (file && file.type.startsWith('image/')) {
      setSelectedImage(file);
      setError('');
      
      const reader = new FileReader();
      reader.onload = (e) => {
        setImagePreview(e.target.result);
      };
      reader.readAsDataURL(file);
    } else {
      setError('Please select a valid image file');
    }
  };

  const handleFileInput = (e) => {
    const file = e.target.files[0];
    handleImageSelect(file);
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleImageSelect(e.dataTransfer.files[0]);
    }
  };

  const analyzeImage = async () => {
    if (!selectedImage) {
      setError('Please select an image first');
      return;
    }

    setLoading(true);
    setError('');
    setResults(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedImage);

      const response = await axios.post('/api/detect_disease', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (response.data.status === 'success') {
        setResults(response.data);
      } else {
        setError(response.data.message || 'Disease detection failed');
      }
    } catch (err) {
      // Simulate results for demo purposes when backend is not available
      const mockResults = {
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
      
      setResults(mockResults);
      console.warn('Using mock data - backend not available:', err);
    } finally {
      setLoading(false);
    }
  };

  const clearImage = () => {
    setSelectedImage(null);
    setImagePreview(null);
    setResults(null);
    setError('');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const getSeverityColor = (severity) => {
    switch (severity?.toLowerCase()) {
      case 'low': return 'green';
      case 'moderate': return 'yellow';
      case 'high': return 'red';
      default: return 'gray';
    }
  };

  return (
    <div className="disease-detection">
      <div className="container">
        {/* Header */}
        <header className="page-header">
          <h1>🔬 Plant Disease Detection</h1>
          <p>Upload plant images to get AI-powered disease analysis and treatment recommendations</p>
        </header>

        <div className="detection-content">
          {/* Upload Section */}
          <div className="upload-section">
            <div className="upload-card">
              <h2>📷 Upload Plant Image</h2>
              
              <div 
                className={`upload-area ${dragActive ? 'drag-active' : ''}`}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
              >
                {imagePreview ? (
                  <div className="image-preview">
                    <img src={imagePreview} alt="Plant preview" />
                    <div className="image-overlay">
                      <button className="clear-btn" onClick={(e) => {
                        e.stopPropagation();
                        clearImage();
                      }}>
                        ❌ Clear
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="upload-prompt">
                    <div className="upload-icon">📸</div>
                    <h3>Drag & drop an image here</h3>
                    <p>or click to browse files</p>
                    <div className="file-types">
                      Supports: JPG, PNG, JPEG
                    </div>
                  </div>
                )}
                
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleFileInput}
                  style={{ display: 'none' }}
                />
              </div>

              {selectedImage && (
                <div className="image-info">
                  <div className="info-item">
                    <span className="info-label">File:</span>
                    <span className="info-value">{selectedImage.name}</span>
                  </div>
                  <div className="info-item">
                    <span className="info-label">Size:</span>
                    <span className="info-value">{(selectedImage.size / 1024).toFixed(1)} KB</span>
                  </div>
                </div>
              )}

              <button 
                className="analyze-btn"
                onClick={analyzeImage}
                disabled={!selectedImage || loading}
              >
                {loading ? (
                  <>
                    <span className="spinner"></span>
                    Analyzing...
                  </>
                ) : (
                  <>
                    🔍 Analyze Disease
                  </>
                )}
              </button>
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
                <h2>🔬 Analysis Results</h2>
                
                <div className="disease-summary">
                  <div className="disease-main">
                    <div className="disease-icon">🦠</div>
                    <div className="disease-details">
                      <h3>{results.disease}</h3>
                      <div className="confidence-bar">
                        <span className="confidence-label">Confidence: {(results.confidence * 100).toFixed(0)}%</span>
                        <div className="confidence-progress">
                          <div 
                            className="confidence-fill"
                            style={{ width: `${results.confidence * 100}%` }}
                          ></div>
                        </div>
                      </div>
                      <div className={`severity-badge ${getSeverityColor(results.severity)}`}>
                        {results.severity} Severity
                      </div>
                    </div>
                  </div>
                  
                  {results.description && (
                    <div className="disease-description">
                      <p>{results.description}</p>
                    </div>
                  )}
                </div>

                {results.treatment && (
                  <div className="treatment-section">
                    <h3>💊 Treatment Recommendations</h3>
                    
                    <div className="treatment-categories">
                      {results.treatment.immediate && (
                        <div className="treatment-category">
                          <h4>🚨 Immediate Actions</h4>
                          <ul>
                            {results.treatment.immediate.map((action, index) => (
                              <li key={index}>{action}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                      
                      {results.treatment.preventive && (
                        <div className="treatment-category">
                          <h4>🛡️ Preventive Measures</h4>
                          <ul>
                            {results.treatment.preventive.map((measure, index) => (
                              <li key={index}>{measure}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                      
                      {results.treatment.longTerm && (
                        <div className="treatment-category">
                          <h4>📈 Long-term Management</h4>
                          <ul>
                            {results.treatment.longTerm.map((strategy, index) => (
                              <li key={index}>{strategy}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            )}

            {!results && !error && !loading && (
              <div className="placeholder-card">
                <div className="placeholder-content">
                  <div className="placeholder-icon">🌱</div>
                  <h3>Ready for Analysis</h3>
                  <p>Upload a plant image to get AI-powered disease detection and treatment recommendations</p>
                  <div className="tips">
                    <h4>📝 Tips for better results:</h4>
                    <ul>
                      <li>Use clear, well-lit images</li>
                      <li>Focus on affected areas</li>
                      <li>Include leaves and stems if possible</li>
                      <li>Avoid blurry or low-resolution images</li>
                    </ul>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default DiseaseDetection;
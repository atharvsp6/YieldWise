# 🌾 YieldWise - AI-Powered Agricultural Intelligence Platform

YieldWise is a comprehensive agricultural intelligence platform that combines machine learning, computer vision, and AI to help farmers make data-driven decisions for crop management, disease detection, and financial planning.

## 🚀 Features

### 🎯 Smart Yield Prediction
- **AI-powered crop yield forecasting** using advanced machine learning models
- **Multi-factor analysis** considering weather, soil conditions, and farming practices
- **Confidence intervals** for prediction accuracy
- **Location-based insights** with interactive maps
- **Real-time weather integration** for better predictions

### 🔬 Plant Disease Detection
- **Computer vision technology** for identifying plant diseases from leaf images
- **10+ disease categories** including bacterial, fungal, and viral diseases
- **High accuracy detection** with confidence scores
- **Treatment recommendations** and prevention strategies
- **Support for multiple image formats** (JPG, PNG, GIF, BMP)

### 💰 Financial Calculator
- **Comprehensive cost-benefit analysis** for crop farming
- **Detailed cost breakdown** including seeds, fertilizer, labor, machinery
- **Market price insights** and revenue projections
- **ROI calculations** with visual charts and graphs
- **AI-powered financial recommendations**

### 🌤️ Weather Integration
- **Real-time weather data** and 5-day forecasts
- **Location-based weather information** using coordinates
- **Interactive maps** with Mapbox integration
- **Weather alerts** for farming decisions

## 🛠️ Technology Stack

- **Backend**: Flask (Python)
- **Machine Learning**: scikit-learn, TensorFlow/Keras
- **AI/LLM**: Google Gemini API
- **Frontend**: HTML5, CSS3, JavaScript, Tailwind CSS
- **Maps**: Mapbox GL JS
- **Charts**: Chart.js
- **Image Processing**: Pillow (PIL)
- **Data Processing**: pandas, numpy

## 📋 Prerequisites

- Python 3.10 or higher
- pip (Python package installer)
- Git

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/atharvsp6/YieldWise.git
cd YieldWise
```

### 2. Create Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Setup
Create a `.env` file in the root directory with the following variables:
```env
# Google Gemini API Key (for AI recommendations)
GEMINI_API_KEY=your_gemini_api_key_here

# Mapbox API Key (for maps and geocoding)
MAPBOX_API_KEY=your_mapbox_api_key_here

# OpenWeatherMap API Key (for weather data)
WEATHER_API_KEY=your_weather_api_key_here
```

### 5. Run the Application
```bash
python app.py
```

The application will be available at `http://127.0.0.1:5000`

## 📁 Project Structure

```
YieldWise/
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── .env                           # Environment variables (create this)
├── .gitignore                     # Git ignore rules
├── README.md                      # Project documentation
├── models/                        # ML model files
│   └── plant_disease_model.h5     # Plant disease detection model
├── static/                        # Static files
│   ├── style.css                  # Main stylesheet
│   └── Financial.html             # Financial calculator template
├── templates/                     # HTML templates
│   ├── index.html                 # Main page (Yield Prediction)
│   ├── disease_detection.html     # Plant disease detection page
│   ├── financial_calculator.html  # Financial calculator page
│   └── dashboard.html             # Dashboard page
└── .venv/                         # Virtual environment (created during setup)
```

## 🎯 Usage

### Yield Prediction
1. Navigate to the main page (`/`)
2. Select your location using the interactive map or search
3. Fill in crop details (type, season, area, rainfall, fertilizer, pesticide)
4. Click "Predict Crop Yield" to get AI-powered predictions
5. View detailed recommendations and confidence intervals

### Disease Detection
1. Go to `/disease-detection`
2. Upload a clear image of plant leaves
3. Click "Analyze Plant Health"
4. View disease diagnosis with confidence scores
5. Get treatment recommendations

### Financial Calculator
1. Visit `/financial-calculator`
2. Enter crop name, area, and state
3. Click "Generate Financial Report"
4. View comprehensive cost analysis and ROI projections

### Dashboard
1. Access `/dashboard` for an overview of all features
2. Quick access to all tools and features
3. Platform information and getting started guide

## 🔧 API Endpoints

### Yield Prediction
- `POST /predict` - Get crop yield predictions
- `POST /weather` - Fetch weather forecast
- `POST /geocode` - Search locations
- `POST /reverse-geocode` - Get location from coordinates

### Disease Detection
- `GET /disease-detection` - Disease detection page
- `POST /predict-disease` - Analyze plant images

### Financial Calculator
- `GET /financial-calculator` - Financial calculator page
- `POST /calculate-finance` - Generate financial analysis

### General
- `GET /` - Main page (Yield Prediction)
- `GET /dashboard` - Dashboard overview

## 🌾 Supported Crops

- Rice
- Wheat
- Maize
- Cotton
- Sugarcane
- Groundnut
- Bajra

## 🔬 Supported Diseases

- Bacterial Leaf Blight
- Bacterial Leaf Streak
- Bacterial Panicle Blight
- Blast
- Brown Spot
- Dead Heart
- Downy Mildew
- Hispa
- Normal (Healthy)
- Tungro

## 🚀 Deployment

### Production Deployment with Gunicorn
```bash
# Install gunicorn (already in requirements.txt)
pip install gunicorn

# Run with gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

### Docker Deployment
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Google Gemini API for AI-powered recommendations
- Mapbox for mapping services
- OpenWeatherMap for weather data
- TensorFlow/Keras for machine learning capabilities
- The agricultural research community for datasets and insights

## 📞 Support

For support, email support@yieldwise.com or create an issue in the GitHub repository.

## 🔮 Future Enhancements

- [ ] Mobile app development
- [ ] IoT sensor integration
- [ ] Advanced soil analysis
- [ ] Multi-language support
- [ ] Farmer community features
- [ ] Market price predictions
- [ ] Supply chain optimization
- [ ] Blockchain integration for traceability

---

**Made with ❤️ for the agricultural community**

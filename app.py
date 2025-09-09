# app.py - Integrated YieldWise Platform with all models

from flask import Flask, request, jsonify, render_template, send_from_directory
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import google.generativeai as genai
import json
import warnings
import requests
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# SMART CROP ADVISOR CLASS
# -----------------------------------------------------------------------------

class SmartCropAdvisor:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.gemini_model = None
        self.data_stats = {}

    def setup_gemini_api(self, api_key):
        """Setup Google Gemini API with enhanced configuration"""
        if not api_key:
            return False, "⚠️ Gemini API key not found. AI recommendations are disabled."
        
        try:
            genai.configure(api_key=api_key)
            self.gemini_model = genai.GenerativeModel(
                'gemini-1.5-flash',
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    top_p=0.8,
                    top_k=40,
                    max_output_tokens=8192,
                )
            )
            self.gemini_model.generate_content("Hello")  # Test call
            return True, "✅ Gemini AI Assistant is configured and ready."
        except Exception as e:
            return False, f"❌ Error setting up Gemini API: {e}"

    def create_sample_data(self):
        """Create realistic sample crop data"""
        np.random.seed(42)
        n_samples = 2000
        crops = ['Rice', 'Wheat', 'Maize', 'Cotton', 'Sugarcane', 'Groundnut', 'Bajra']
        seasons = ['Kharif', 'Rabi', 'Summer', 'Whole Year']
        states = ['Andhra Pradesh', 'Karnataka', 'Tamil Nadu', 'Kerala', 'Maharashtra',
                  'Punjab', 'Haryana', 'Gujarat', 'Rajasthan', 'West Bengal']

        data = []
        for i in range(n_samples):
            crop = np.random.choice(crops)
            season = np.random.choice(seasons)
            state = np.random.choice(states)
            area = np.random.uniform(10, 500)
            rainfall = np.random.uniform(300, 2500)
            fertilizer = np.random.uniform(20, 150)
            pesticide = np.random.uniform(5, 40)

            base_yield = {'Rice': 3.5, 'Wheat': 3.2, 'Maize': 4.8, 'Cotton': 1.8,
                          'Sugarcane': 65, 'Groundnut': 1.4, 'Bajra': 1.8}[crop]
            season_factor = {'Kharif': 1.1, 'Rabi': 1.0, 'Summer': 0.9, 'Whole Year': 1.2}[season]
            rain_factor = min(1.3, 0.5 + rainfall/1500) if crop in ['Rice', 'Sugarcane'] else min(1.2, 0.6 + rainfall/2000)
            fert_factor = min(1.4, 0.7 + (fertilizer/100) * 0.8)
            yield_value = (base_yield * season_factor * rain_factor * fert_factor * np.random.uniform(0.85, 1.15))

            data.append({'Crop': crop, 'Crop_Year': np.random.randint(2018, 2024), 'Season': season,
                         'State': state, 'Area': round(area, 1), 'Annual_Rainfall': round(rainfall, 1),
                         'Fertilizer': round(fertilizer, 1), 'Pesticide': round(pesticide, 1),
                         'Yield': round(yield_value, 2)})

        return pd.DataFrame(data)

    def load_and_preprocess_data(self, df=None):
        """Load and preprocess crop data"""
        if df is None:
            df = self.create_sample_data()

        self.data_stats = {
            'crops': df['Crop'].unique().tolist(),
            'seasons': df['Season'].unique().tolist(),
            'states': df['State'].unique().tolist(),
            'fertilizer_range': [df['Fertilizer'].min(), df['Fertilizer'].max()],
            'rainfall_range': [df['Annual_Rainfall'].min(), df['Annual_Rainfall'].max()],
            'yield_by_crop': df.groupby('Crop')['Yield'].agg(['mean', 'std']).to_dict()
        }

        df['Fertilizer_per_Area'] = df['Fertilizer'] / (df['Area'] + 1)
        df['Pesticide_per_Area'] = df['Pesticide'] / (df['Area'] + 1)
        df['Rainfall_Category'] = pd.cut(df['Annual_Rainfall'], bins=[0, 800, 1500, 3000], labels=['Low', 'Medium', 'High'])

        categorical_cols = ['Crop', 'Season', 'State', 'Rainfall_Category']
        for col in categorical_cols:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le

        self.feature_names = ['Crop_encoded', 'Crop_Year', 'Season_encoded', 'State_encoded', 'Area',
                              'Annual_Rainfall', 'Fertilizer_per_Area', 'Pesticide_per_Area', 'Rainfall_Category_encoded']

        X = df[self.feature_names]
        y = df['Yield']
        X_scaled = self.scaler.fit_transform(X)

        return X_scaled, y

    def train_model(self, X, y):
        """Train enhanced Random Forest model"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=5,
                                             min_samples_leaf=2, random_state=42, n_jobs=-1)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        return r2, rmse

    def predict_yield(self, crop, season, state, area, rainfall, fertilizer, pesticide, year=2024):
        """Enhanced yield prediction with confidence intervals"""
        if self.model is None:
            return None, None, "Model not trained yet!"

        try:
            input_data = pd.DataFrame({'Crop': [crop], 'Crop_Year': [year], 'Season': [season], 'State': [state],
                                       'Area': [area], 'Annual_Rainfall': [rainfall], 'Fertilizer': [fertilizer],
                                       'Pesticide': [pesticide]})

            input_data['Fertilizer_per_Area'] = input_data['Fertilizer'] / (input_data['Area'] + 1)
            input_data['Pesticide_per_Area'] = input_data['Pesticide'] / (input_data['Area'] + 1)
            input_data['Rainfall_Category'] = pd.cut(input_data['Annual_Rainfall'], bins=[0, 800, 1500, 3000], labels=['Low', 'Medium', 'High'])

            for col in ['Crop', 'Season', 'State', 'Rainfall_Category']:
                if col in self.label_encoders:
                    try:
                        input_data[f'{col}_encoded'] = self.label_encoders[col].transform(input_data[col])
                    except ValueError:
                        input_data[f'{col}_encoded'] = 0
                else:
                    input_data[f'{col}_encoded'] = 0

            X_input = input_data[self.feature_names]
            X_input_scaled = self.scaler.transform(X_input)

            predictions = [tree.predict(X_input_scaled)[0] for tree in self.model.estimators_]
            predicted_yield = np.mean(predictions)
            confidence_interval = np.percentile(predictions, [25, 75])

            return predicted_yield, confidence_interval.tolist(), "Success"

        except Exception as e:
            return None, None, f"Prediction error: {str(e)}"

    def get_comprehensive_recommendations(self, crop, predicted_yield, current_conditions, confidence_interval):
        """Get comprehensive recommendations"""
        if self.gemini_model is None:
            return None  # Return None if not configured

        try:
            crop_stats = self.data_stats['yield_by_crop']['mean'].get(crop, 3.0)
            
            if predicted_yield > crop_stats * 1.2:
                yield_category = "High"
            elif predicted_yield > crop_stats * 0.8:
                yield_category = "Medium"
            else:
                yield_category = "Low"

            prompt = f"""
You are an expert agricultural scientist. Provide SPECIFIC, CONCISE recommendations in a structured JSON format. Avoid conversational text.

Analyze the following data for a farmer:

- **Crop:** {crop}
- **Predicted Yield:** {predicted_yield:.2f} tons/hectare (Confidence: {confidence_interval[0]:.2f}-{confidence_interval[1]:.2f})
- **Yield Category:** {yield_category} (Average is {crop_stats:.2f})
- **Conditions:** State: {current_conditions['state']}, Season: {current_conditions['season']}, Rainfall: {current_conditions['rainfall']} mm, Fertilizer: {current_conditions['fertilizer']} kg/ha.

Return a single JSON object with these exact keys: "yield_assessment", "fertilizer_management", "irrigation_plan", "planting_strategy", "risk_mitigation","cost-benefit_analysis".

For each key, provide a dictionary of actionable advice. Example for fertilizer_management: {{"npk_ratio": "...", "application_timing": "...", "organic_options": "..."}}, cost-benefit_analysis: {{"expected_yield_increase": "...", "cost": "...", "benefit": "...", "roi_estimate": "..."}}. Be direct and practical.
"""

            response = self.gemini_model.generate_content(prompt)
            response_text = response.text.strip().replace('```json', '').replace('```', '')
            return json.loads(response_text)

        except Exception as e:
            return {"error": f"LLM Error: {str(e)}", "raw_response": response.text if 'response' in locals() else "No response from model."}

# -----------------------------------------------------------------------------
# PLANT DISEASE DETECTION CLASS
# -----------------------------------------------------------------------------

class PlantDiseaseDetector:
    def __init__(self):
        self.model = None
        self.class_labels = ['bacterial_leaf_blight', 'bacterial_leaf_streak', 'bacterial_panicle_blight',
                           'blast', 'brown_spot', 'dead_heart', 'downy_mildew', 'hispa', 'normal', 'tungro']
        self.model_path = 'models/plant_disease_model.h5'
        self.load_model()
    
    def load_model(self):
        """Load the plant disease detection model"""
        try:
            if os.path.exists(self.model_path):
                self.model = load_model(self.model_path)
                print("✅ Plant Disease Detection Model loaded successfully")
            else:
                print("⚠️ Plant Disease Detection Model not found. Creating placeholder model.")
                # Create a simple placeholder model for demonstration
                self.model = self.create_placeholder_model()
        except Exception as e:
            print(f"❌ Error loading plant disease model: {e}")
            self.model = self.create_placeholder_model()
    
    def create_placeholder_model(self):
        """Create a placeholder model for demonstration purposes"""
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
        
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(len(self.class_labels), activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        return model
    
    def predict_disease(self, image_file):
        """Predict plant disease from uploaded image"""
        try:
            # Read and preprocess image
            img = Image.open(image_file)
            img = img.convert('RGB')
            img = img.resize((128, 128))
            
            # Convert to array and normalize
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Make prediction
            predictions = self.model.predict(img_array)
            predicted_class_idx = np.argmax(predictions[0])
            predicted_class = self.class_labels[predicted_class_idx]
            confidence = float(np.max(predictions[0]) * 100)
            
            return {
                'disease': predicted_class,
                'confidence': round(confidence, 2),
                'all_predictions': {
                    label: float(prob * 100) 
                    for label, prob in zip(self.class_labels, predictions[0])
                }
            }
        except Exception as e:
            return {'error': f'Prediction failed: {str(e)}'}

# -----------------------------------------------------------------------------
# FINANCIAL CALCULATOR CLASS
# -----------------------------------------------------------------------------

class FinancialCalculator:
    def __init__(self):
        self.gemini_model = None
        self.setup_gemini_api()
    
    def setup_gemini_api(self):
        """Setup Google Gemini API for financial calculations"""
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            try:
                genai.configure(api_key=api_key)
                self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                print("✅ Financial Calculator AI configured")
            except Exception as e:
                print(f"⚠️ Financial Calculator AI setup failed: {e}")
    
    def calculate_financial_analysis(self, crop, area, state):
        """Calculate financial analysis for crop farming"""
        if not self.gemini_model:
            return self.get_sample_financial_data(crop, area, state)
        
        try:
            prompt = f"""
            You are an expert agricultural financial advisor for the Indian market. 
            Calculate a detailed financial breakdown for growing {crop} on {area} acres of land in {state}, India.
            
            Provide your response as a JSON object with the following structure:
            {{
                "estimated_costs": {{
                    "seeds": {{"total_cost": number, "justification": "string"}},
                    "fertilizer": {{"total_cost": number, "justification": "string"}},
                    "pesticides_herbicides": {{"total_cost": number, "justification": "string"}},
                    "labor": {{"total_cost": number, "justification": "string"}},
                    "machinery_fuel": {{"total_cost": number, "justification": "string"}},
                    "irrigation": {{"total_cost": number, "justification": "string"}},
                    "miscellaneous": {{"total_cost": number, "justification": "string"}}
                }},
                "total_expenditure": number,
                "market_analysis": {{
                    "average_yield_per_acre": "string",
                    "average_market_price": "string",
                    "estimated_revenue": number,
                    "justification": "string"
                }},
                "profit_analysis": {{
                    "potential_profit": number,
                    "return_on_investment": "string"
                }},
                "summary": "string"
            }}
            
            Base calculations on current Indian market rates and MSP where applicable.
            All amounts in Indian Rupees.
            """
            
            response = self.gemini_model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Clean up response text
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            return json.loads(response_text)
        except Exception as e:
            return self.get_sample_financial_data(crop, area, state)
    
    def get_sample_financial_data(self, crop, area, state):
        """Provide sample financial data when AI is not available"""
        area_num = float(area)
        
        # Sample cost calculations (simplified)
        costs = {
            "seeds": {"total_cost": int(area_num * 2000), "justification": f"Based on {crop} seed rates for {area} acres"},
            "fertilizer": {"total_cost": int(area_num * 3000), "justification": f"NPK fertilizers for {area} acres"},
            "pesticides_herbicides": {"total_cost": int(area_num * 1500), "justification": f"Pest control for {area} acres"},
            "labor": {"total_cost": int(area_num * 5000), "justification": f"Manual labor costs for {area} acres"},
            "machinery_fuel": {"total_cost": int(area_num * 2000), "justification": f"Tractor and fuel costs"},
            "irrigation": {"total_cost": int(area_num * 1000), "justification": f"Water and irrigation costs"},
            "miscellaneous": {"total_cost": int(area_num * 1000), "justification": f"Other farming expenses"}
        }
        
        total_expenditure = sum(cost["total_cost"] for cost in costs.values())
        estimated_revenue = int(area_num * 15000)  # Sample revenue calculation
        potential_profit = estimated_revenue - total_expenditure
        roi = f"{(potential_profit/total_expenditure)*100:.1f}%" if total_expenditure > 0 else "0%"
        
        return {
            "estimated_costs": costs,
            "total_expenditure": total_expenditure,
            "market_analysis": {
                "average_yield_per_acre": f"{area_num * 2:.1f} tons",
                "average_market_price": "₹7,500/ton",
                "estimated_revenue": estimated_revenue,
                "justification": f"Based on average market rates for {crop} in {state}"
            },
            "profit_analysis": {
                "potential_profit": potential_profit,
                "return_on_investment": roi
            },
            "summary": f"Sample financial analysis for {crop} farming in {state}. Actual results may vary based on market conditions."
        }

# -----------------------------------------------------------------------------
# FLASK APP SETUP
# -----------------------------------------------------------------------------

load_dotenv()  # Load environment variables from .env file
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size


# Initialize all models
print("Initializing YieldWise Platform...")
advisor = SmartCropAdvisor()
disease_detector = PlantDiseaseDetector()
financial_calculator = FinancialCalculator()

# Setup Gemini API on startup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
api_is_ok, api_startup_message = advisor.setup_gemini_api(GEMINI_API_KEY)
print(api_startup_message)  # Print status to the console

# Train the ML model
X, y = advisor.load_and_preprocess_data()
r2, rmse = advisor.train_model(X, y)
print(f"✅ ML Model trained and ready. Performance: R²={r2:.3f}, RMSE={rmse:.3f}")

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)


@app.route('/')
def index():
    """Render the main HTML page and pass the Mapbox API key."""
    mapbox_api_key = os.getenv("MAPBOX_API_KEY")
    return render_template('index.html', mapbox_key=mapbox_api_key)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle the prediction request."""
    data = request.json
    
    try:
        predicted_yield, confidence, status = advisor.predict_yield(
            crop=data['crop'],
            season=data['season'],
            state=data['state'],
            area=float(data['area']),
            rainfall=float(data['rainfall']),
            fertilizer=float(data['fertilizer']),
            pesticide=float(data['pesticide'])
        )

        if predicted_yield is None:
            return jsonify({'error': status}), 400

        # Get AI recommendations (will be None if API is not configured)
        current_conditions = {
            'state': data['state'], 'season': data['season'],
            'rainfall': data['rainfall'], 'fertilizer': data['fertilizer']
        }

        recommendations = advisor.get_comprehensive_recommendations(
            data['crop'], predicted_yield, current_conditions, confidence
        )

        response = {
            'predicted_yield': round(predicted_yield, 2),
            'confidence_interval': [round(c, 2) for c in confidence],
            'total_production': round(predicted_yield * float(data['area']), 2),
            'recommendations': recommendations
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': f"An unexpected error occurred: {str(e)}"}), 500

@app.route('/weather', methods=['POST'])
def get_weather():
    """Fetch live weather forecast using coordinates."""
    data = request.json
    lat = data.get('lat')
    lon = data.get('lon')
    
    if not lat or not lon:
        return jsonify({'error': 'Latitude and longitude coordinates are required.'}), 400

    WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
    if not WEATHER_API_KEY:
        return jsonify({'error': 'Weather API key not configured on server.'}), 500

    # API endpoint for 5-day/3-hour forecast using coordinates
    url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"

    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        weather_data = response.json()

        # Process the data to be more frontend-friendly
        forecast = []
        # Get one forecast per day (OpenWeatherMap gives data every 3 hours)
        for i in range(0, min(len(weather_data['list']), 40), 8):  # 5 days max
            day_data = weather_data['list'][i]
            forecast.append({
                'date': day_data['dt_txt'].split(' ')[0],
                'temp': round(day_data['main']['temp']),
                'description': day_data['weather'][0]['description'].title(),
                'icon': day_data['weather'][0]['icon'],
                'humidity': day_data['main']['humidity'],
                'wind_speed': day_data['wind']['speed']
            })

        return jsonify({'forecast': forecast, 'location': weather_data['city']['name']})

    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Failed to fetch weather data: {e}'}), 500

@app.route('/geocode', methods=['POST'])
def geocode_location():
    """Geocode location using Mapbox API."""
    data = request.json
    query = data.get('query')  # This can be city name, pincode, address, etc.
    
    if not query:
        return jsonify({'error': 'Search query is required.'}), 400

    MAPBOX_API_KEY = os.getenv("MAPBOX_API_KEY")
    if not MAPBOX_API_KEY:
        return jsonify({'error': 'Mapbox API key not configured on server.'}), 500

    # Use Mapbox Geocoding API to search for locations
    url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{query}.json?access_token={MAPBOX_API_KEY}&country=IN&limit=5"

    try:
        response = requests.get(url)
        response.raise_for_status()
        geocode_data = response.json()

        if geocode_data['features']:
            results = []
            for feature in geocode_data['features']:
                results.append({
                    'place_name': feature['place_name'],
                    'coordinates': feature['center'],  # [longitude, latitude]
                    'context': feature.get('context', [])
                })
            return jsonify({'results': results})
        else:
            return jsonify({'error': 'No locations found for the given query.'}), 404

    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Failed to geocode location: {e}'}), 500

@app.route('/reverse-geocode', methods=['POST'])
def reverse_geocode():
    """Reverse geocode coordinates to get location name and extract state."""
    data = request.json
    lat = data.get('lat')
    lon = data.get('lon')
    
    if not lat or not lon:
        return jsonify({'error': 'Latitude and longitude are required.'}), 400

    MAPBOX_API_KEY = os.getenv("MAPBOX_API_KEY")
    if not MAPBOX_API_KEY:
        return jsonify({'error': 'Mapbox API key not configured on server.'}), 500

    # Use Mapbox Reverse Geocoding API
    # Added types for better context filtering: region for state, place for city, country for country
    url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{lon},{lat}.json?access_token={MAPBOX_API_KEY}&country=IN&types=place,region,country"

    try:
        response = requests.get(url)
        response.raise_for_status()
        geocode_data = response.json()

        place_name = 'Unknown Location'
        state_name = 'Unknown State'
        
        if geocode_data['features']:
            # Prioritize a more detailed place_name if available
            place_name = geocode_data['features'][0]['place_name']
            
            # Iterate through context to find the state (region type)
            for component in geocode_data['features'][0]['context']:
                if 'id' in component and component['id'].startswith('region.'):
                    state_name = component['text']
                    break
            # Fallback if state not found in context directly (sometimes in primary place_name for smaller places)
            if state_name == 'Unknown State':
                for component in geocode_data['features'][0]['properties'].get('short_code', '').split(','):
                    if component.startswith('IN-'): # Indian state code prefix
                        state_name = component[3:] # Remove 'IN-' prefix
                        break
                if state_name == 'Unknown State':
                     # Last resort: try to extract from place_name if it contains common state names
                     # This is a heuristic and might not be perfect
                    common_indian_states = ['Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh', 'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jharkhand', 'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal']
                    for state in common_indian_states:
                        if state in place_name:
                            state_name = state
                            break

        return jsonify({
            'place_name': place_name,
            'coordinates': [lat, lon],
            'state': state_name # Return state explicitly
        })
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Failed to reverse geocode: {e}'}), 500

# -----------------------------------------------------------------------------
# PLANT DISEASE DETECTION ROUTES
# -----------------------------------------------------------------------------

@app.route('/disease-detection')
def disease_detection_page():
    """Render the plant disease detection page"""
    return render_template('disease_detection.html')

@app.route('/predict-disease', methods=['POST'])
def predict_disease():
    """Handle plant disease prediction from uploaded image"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            # Make prediction
            result = disease_detector.predict_disease(file)
            
            if 'error' in result:
                return jsonify(result), 500
            
            return jsonify(result)
        else:
            return jsonify({'error': 'Invalid file type. Please upload an image file.'}), 400
    
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

def allowed_file(filename):
    """Check if file extension is allowed"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# -----------------------------------------------------------------------------
# FINANCIAL CALCULATOR ROUTES
# -----------------------------------------------------------------------------

@app.route('/financial-calculator')
def financial_calculator_page():
    """Render the financial calculator page"""
    return render_template('financial_calculator.html')

@app.route('/calculate-finance', methods=['POST'])
def calculate_finance():
    """Handle financial calculation request"""
    try:
        data = request.json
        crop = data.get('crop')
        area = data.get('area')
        state = data.get('state')
        
        if not all([crop, area, state]):
            return jsonify({'error': 'Missing required fields: crop, area, state'}), 400
        
        # Calculate financial analysis
        result = financial_calculator.calculate_financial_analysis(crop, area, state)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': f'Calculation failed: {str(e)}'}), 500

# -----------------------------------------------------------------------------
# DASHBOARD ROUTE
# -----------------------------------------------------------------------------

@app.route('/dashboard')
def dashboard():
    """Render the main dashboard with all features"""
    return render_template('dashboard.html')

# -----------------------------------------------------------------------------
# STATIC FILE SERVING
# -----------------------------------------------------------------------------

@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files"""
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True)
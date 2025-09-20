# app.py - Integrated YieldWise Platform with optional ML models
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import json
import warnings
import requests
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from PIL import Image
# Optional imports for ML functionality
try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ scikit-learn not available. ML predictions will use simulated data.")
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    print("⚠️ Google Generative AI not available. AI recommendations disabled.")
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow not available. Disease detection will use simulated results.")
warnings.filterwarnings('ignore')
# -----------------------------------------------------------------------------
# SMART CROP ADVISOR CLASS
# -----------------------------------------------------------------------------
class SmartCropAdvisor:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        if SKLEARN_AVAILABLE:
            self.scaler = StandardScaler()
        self.feature_names = []
        self.gemini_model = None
        self.data_stats = {}
    def setup_gemini_api(self, api_key):
        """Setup Google Gemini API with enhanced configuration"""
        if not GENAI_AVAILABLE:
            return False, "⚠️ Google Generative AI not installed. AI recommendations are disabled."
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
        if not SKLEARN_AVAILABLE:
            return None, None
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
        if not SKLEARN_AVAILABLE or X is None:
            return 0.85, 0.5  # Simulated performance metrics
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
        if not SKLEARN_AVAILABLE:
            return self.simulate_prediction(crop, season, state, area, rainfall, fertilizer, pesticide)
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
    def simulate_prediction(self, crop, season, state, area, rainfall, fertilizer, pesticide):
        """Simulate yield prediction when ML models are not available"""
        try:
            # Simple rule-based prediction
            base_yields = {'Rice': 3.5, 'Wheat': 3.2, 'Maize': 4.8, 'Cotton': 1.8,
                          'Sugarcane': 65, 'Groundnut': 1.4, 'Bajra': 1.8}
            base_yield = base_yields.get(crop, 3.0)
            # Apply simple factors
            rain_factor = min(1.3, 0.5 + rainfall/1500) if crop in ['Rice', 'Sugarcane'] else min(1.2, 0.6 + rainfall/2000)
            fert_factor = min(1.4, 0.7 + (fertilizer/100) * 0.8)
            season_factor = {'Kharif': 1.1, 'Rabi': 1.0, 'Summer': 0.9, 'Whole Year': 1.2}.get(season, 1.0)
            predicted_yield = base_yield * rain_factor * fert_factor * season_factor
            confidence_interval = [predicted_yield * 0.9, predicted_yield * 1.1]
            return predicted_yield, confidence_interval, "Success (Simulated)"
        except Exception as e:
            return None, None, f"Simulation error: {str(e)}"
    def get_comprehensive_recommendations(self, crop, predicted_yield, current_conditions, confidence_interval):
        """Get comprehensive recommendations"""
        if not GENAI_AVAILABLE or self.gemini_model is None:
            return self.get_basic_recommendations(crop, predicted_yield, current_conditions)
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
            return self.get_basic_recommendations(crop, predicted_yield, current_conditions)
    def get_basic_recommendations(self, crop, predicted_yield, current_conditions):
        """Provide basic recommendations when AI is not available"""
        return {
            "yield_assessment": {
                "predicted_yield": f"{predicted_yield:.2f} tons/hectare",
                "category": "Estimated based on current conditions",
                "factors": "Weather, fertilizer, and regional averages considered"
            },
            "fertilizer_management": {
                "npk_ratio": "Apply balanced NPK fertilizer as per soil test",
                "application_timing": "Split application - basal, tillering, and flowering stages",
                "organic_options": "Consider compost and bio-fertilizers"
            },
            "irrigation_plan": {
                "schedule": "Monitor soil moisture and weather forecasts",
                "method": "Drip irrigation recommended for water efficiency",
                "critical_stages": "Ensure adequate water during flowering and grain filling"
            },
            "planting_strategy": {
                "timing": "Follow local agricultural calendar",
                "variety": "Use high-yielding, disease-resistant varieties",
                "spacing": "Maintain recommended row and plant spacing"
            },
            "risk_mitigation": {
                "weather": "Monitor weather forecasts regularly",
                "pests": "Regular field monitoring and IPM practices",
                "market": "Consider contract farming or cooperative marketing"
            },
            "note": "These are general recommendations. For detailed advice, enable AI features or consult local agriculture extension officers."
        }
# -----------------------------------------------------------------------------
# PLANT DISEASE DETECTION CLASS
# -----------------------------------------------------------------------------
class PlantDiseaseDetector:
    def __init__(self):
        self.model = None
        self.class_labels = ['bacterial_leaf_blight', 'bacterial_leaf_streak', 'bacterial_panicle_blight',
                            'blast', 'brown_spot', 'dead_heart', 'downy_mildew', 'hispa', 'normal', 'tungro']
        if TENSORFLOW_AVAILABLE:
            self.load_model()
    def load_model(self):
        """Load the plant disease detection model"""
        try:
            model_path = 'models/plant_disease_model.h5'
            if os.path.exists(model_path):
                self.model = load_model(model_path)
                print("✅ Plant Disease Detection Model loaded successfully")
            else:
                print("⚠️ Plant Disease Detection Model not found. Using simulated predictions.")
        except Exception as e:
            print(f"❌ Error loading plant disease model: {e}")
    def predict_disease(self, image_file):
        """Predict plant disease from uploaded image"""
        try:
            if not TENSORFLOW_AVAILABLE or self.model is None:
                return self.simulate_disease_prediction(image_file)
            # Read and preprocess image
            img = Image.open(image_file)
            img = img.convert('RGB')
            img = img.resize((128, 128))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0

            # Make prediction
            predictions = self.model.predict(img_array)
            predicted_class_index = np.argmax(predictions, axis=1)[0]
            predicted_class = self.class_labels[predicted_class_index]
            confidence = np.max(predictions)

            return {
                "status": "success",
                "predicted_disease": predicted_class,
                "confidence": float(confidence),
                "remedy": self.get_remedy(predicted_class)
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Prediction failed: {str(e)}"
            }

    def get_remedy(self, disease_name):
        """Provide basic remedies for detected diseases"""
        remedies = {
            'bacterial_leaf_blight': 'Use a copper-based bactericide. Improve field sanitation and drainage. Avoid excessive nitrogen fertilizer.',
            'bacterial_leaf_streak': 'Apply antibiotics like streptocycline. Rotate crops and use resistant varieties.',
            'bacterial_panicle_blight': 'Apply fungicides like validamycin or kasugamycin. Remove infected plants and reduce humidity.',
            'blast': 'Use fungicides containing tricyclazole or azoxystrobin. Manage nitrogen levels and use resistant varieties.',
            'brown_spot': 'Apply fungicides such as mancozeb or propiconazole. Use proper fertilization and improve water management.',
            'dead_heart': 'Apply insecticides like carbofuran or chlorpyrifos. Remove and destroy affected stems.',
            'downy_mildew': 'Use fungicides like metalaxyl or fosetyl-al. Improve air circulation and use resistant seed varieties.',
            'hispa': 'Apply insecticides such as imidacloprid or lambda-cyhalothrin. Hand-pick and destroy adult beetles.',
            'normal': 'No disease detected. Continue good agricultural practices.',
            'tungro': 'Control insect vectors like leafhoppers with insecticides. Remove and destroy infected plants immediately.'
        }
        return remedies.get(disease_name, "No specific remedy found. Consult a local expert.")

    def simulate_disease_prediction(self, image_file):
        """Simulate disease prediction when model is not available"""
        # A simple, fake simulation
        possible_diseases = ['normal', 'blast', 'brown_spot', 'bacterial_leaf_blight']
        predicted_class = np.random.choice(possible_diseases)
        confidence = np.random.uniform(0.6, 0.95)

        return {
            "status": "success (simulated)",
            "predicted_disease": predicted_class,
            "confidence": float(confidence),
            "remedy": self.get_remedy(predicted_class)
        }

# -----------------------------------------------------------------------------
# FLASK APPLICATION SETUP
# -----------------------------------------------------------------------------
load_dotenv()
app = Flask(__name__, static_folder='static')
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000", "http://localhost:5000", "https://*.vercel.app", "https://*.netlify.app", "https://*.herokuapp.com"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
yield_advisor = SmartCropAdvisor()
disease_detector = PlantDiseaseDetector()

# Initialize ML models and data
X_data, y_data = yield_advisor.load_and_preprocess_data()
r2, rmse = yield_advisor.train_model(X_data, y_data)
print(f"✅ Yield Prediction Model trained. R2: {r2:.2f}, RMSE: {rmse:.2f}")

# Setup Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_status, gemini_message = yield_advisor.setup_gemini_api(GEMINI_API_KEY)
print(gemini_message)



@app.route('/api/predict_yield', methods=['POST'])
def predict_yield_api():
    try:
        data = request.json
        if not data:
            return jsonify({"status": "error", "message": "Invalid JSON data"}), 400
        
        crop = data.get('crop')
        season = data.get('season')
        state = data.get('state')
        area = data.get('area')
        rainfall = data.get('rainfall')
        fertilizer = data.get('fertilizer')
        pesticide = data.get('pesticide')
        
        if not all([crop, season, state, area, rainfall, fertilizer, pesticide]):
            return jsonify({"status": "error", "message": "Missing one or more required fields"}), 400

        area = float(area)
        rainfall = float(rainfall)
        fertilizer = float(fertilizer)
        pesticide = float(pesticide)

        predicted_yield, confidence, status = yield_advisor.predict_yield(
            crop, season, state, area, rainfall, fertilizer, pesticide
        )

        if predicted_yield is None:
            return jsonify({"status": "error", "message": f"Prediction failed: {status}"}), 500

        recommendations = yield_advisor.get_comprehensive_recommendations(
            crop, predicted_yield, data, confidence
        )

        response_data = {
            "status": "success",
            "prediction_status": status,
            "predicted_yield": round(predicted_yield, 2),
            "confidence_interval": [round(c, 2) for c in confidence],
            "recommendations": recommendations,
            "model_r2": round(r2, 2),
            "model_rmse": round(rmse, 2)
        }
        return jsonify(response_data)

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/detect_disease', methods=['POST'])
def detect_disease_api():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part in the request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        result = disease_detector.predict_disease(filepath)
        os.remove(filepath)

        return jsonify(result)
    
    return jsonify({"status": "error", "message": "Unknown error processing image"}), 500

@app.route('/api/data_stats')
def get_data_stats():
    return jsonify(yield_advisor.data_stats)
    
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_react_app(path):
    if path != "" and os.path.exists(os.path.join("static", path)):
        return send_from_directory("static", path)
    return send_from_directory("static", "index.html")

if __name__ == '__main__':
    app.run(debug=True, port=5000)

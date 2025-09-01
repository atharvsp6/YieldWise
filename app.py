# app.py

from flask import Flask, request, jsonify, render_template
import os # NEW: Import the os module
from dotenv import load_dotenv # NEW: Import dotenv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import google.generativeai as genai
import json
import warnings
warnings.filterwarnings('ignore')

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
        try:
            genai.configure(api_key=api_key)
            self.gemini_model = genai.GenerativeModel(
                'gemini-2.5-flash',
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    top_p=0.8,
                    top_k=40,
                    max_output_tokens=9048,
                )
            )
            # Test the API with a simple call
            test_response = self.gemini_model.generate_content("Hello, are you working?")
            print("✅ Gemini API configured successfully!")
            return True
        except Exception as e:
            print(f"❌ Error setting up Gemini API: {e}")
            print("💡 Get your free API key from: https://ai.google.dev/")
            return False

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
                         'State': state, 'Area': round(area, 1), 'Annual_Rainfall': round(rainfall, 1), # <-- THE FIX IS HERE
                         'Fertilizer': round(fertilizer, 1), 'Pesticide': round(pesticide, 1),
                         'Yield': round(yield_value, 2)})
        return pd.DataFrame(data)

    def load_and_preprocess_data(self, df=None):
        """Load and preprocess crop data"""
        if df is None:
            df = self.create_sample_data()
    # ...existing code...

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
            return predicted_yield, confidence_interval, "Success"
        except Exception as e:
            return None, None, f"Prediction error: {str(e)}"

    def get_comprehensive_recommendations(self, crop, predicted_yield, current_conditions, confidence_interval):
        """Get comprehensive recommendations focusing on fertilizer, irrigation, and planting practices"""
        if self.gemini_model is None:
            return "❌ Gemini API not configured. Please provide API key."
        try:
            crop_stats = self.data_stats['yield_by_crop']['mean'].get(crop, 3.0)
            if predicted_yield > crop_stats * 1.2: yield_category = "High"
            elif predicted_yield > crop_stats * 0.8: yield_category = "Medium"
            else: yield_category = "Low"
            prompt = f"""
            Please provide SPECIFIC, CONCISE recommendations using bullet points where possible. Avoid long paragraphs.
            You are an expert agricultural scientist. Analyze the following and provide actionable recommendations.

            CROP ANALYSIS:
            - Crop: {crop}, Predicted Yield: {predicted_yield:.2f} tons/hectare
            - Yield Category: {yield_category} (compared to average {crop_stats:.2f})
            - Confidence Range: {confidence_interval[0]:.2f} - {confidence_interval[1]:.2f} tons/hectare
            CURRENT CONDITIONS:
            - Area: {current_conditions['area']} ha, Rainfall: {current_conditions['rainfall']} mm
            - Fertilizer: {current_conditions['fertilizer']} kg/ha, Pesticide: {current_conditions['pesticide']} kg/ha
            - Season: {current_conditions['season']}, State: {current_conditions['state']}

            Provide recommendations in a structured JSON with these exact keys: "yield_assessment", "fertilizer_recommendations", "irrigation_recommendations", "planting_recommendations", "improvement_potential", "cost_benefit".
            Focus on:
            1. FERTILIZER MANAGEMENT: Optimal NPK ratios, application timing, organic options, micronutrients.
            2. IRRIGATION PRACTICES: Frequency, critical stages, efficient methods, moisture management.
            3. PLANTING PRACTICES: Best dates, seed varieties, spacing, soil preparation.
            4. YIELD IMPROVEMENT: Expected increase, timeline, priority actions, investment.
            5. COST-BENEFIT: ROI estimate, payback period, risk factors.
            """
            response = self.gemini_model.generate_content(prompt)
            try:
                response_text = response.text.strip().replace('```json', '').replace('```', '')
                return json.loads(response_text)
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                return {"yield_assessment": "Analysis completed", "raw_response": response.text, "note": "Response parsing failed, showing raw text"}
        except Exception as e:
            return f"❌ LLM Error: {str(e)}"

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
            self.gemini_model.generate_content("Hello") # Test call
            return True, "✅ Gemini AI Assistant is configured and ready."
        except Exception as e:
            return False, f"❌ Error setting up Gemini API: {e}"

    # ... (the rest of the class methods: create_sample_data, load_and_preprocess_data, etc. are unchanged)
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
        if self.model is None: return None, None, "Model not trained yet!"
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
                    except ValueError: input_data[f'{col}_encoded'] = 0
                else: input_data[f'{col}_encoded'] = 0
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
        if self.gemini_model is None: return None # Return None if not configured
        try:
            crop_stats = self.data_stats['yield_by_crop']['mean'].get(crop, 3.0)
            if predicted_yield > crop_stats * 1.2: yield_category = "High"
            elif predicted_yield > crop_stats * 0.8: yield_category = "Medium"
            else: yield_category = "Low"
            
            prompt = f"""
            You are an expert agricultural scientist. Provide SPECIFIC, CONCISE recommendations in a structured JSON format. Avoid conversational text.
            
            Analyze the following data for a farmer:
            - **Crop:** {crop}
            - **Predicted Yield:** {predicted_yield:.2f} tons/hectare (Confidence: {confidence_interval[0]:.2f}-{confidence_interval[1]:.2f})
            - **Yield Category:** {yield_category} (Average is {crop_stats:.2f})
            - **Conditions:** State: {current_conditions['state']}, Season: {current_conditions['season']}, Rainfall: {current_conditions['rainfall']} mm, Fertilizer: {current_conditions['fertilizer']} kg/ha.

            Return a single JSON object with these exact keys: "yield_assessment", "fertilizer_management", "irrigation_plan", "planting_strategy", "risk_mitigation","cost-benefit analysis".
            
            For each key, provide a dictionary of actionable advice. Example for fertilizer_management: {{"npk_ratio": "...", "application_timing": "...", "organic_options": "..."}}. Be direct and practical.
            """
            
            response = self.gemini_model.generate_content(prompt)
            response_text = response.text.strip().replace('```json', '').replace('```', '')
            return json.loads(response_text)
        except Exception as e:
            return {"error": f"LLM Error: {str(e)}", "raw_response": response.text if 'response' in locals() else "No response from model."}

# -----------------------------------------------------------------------------
# 2. FLASK APP SETUP
# -----------------------------------------------------------------------------
load_dotenv() # NEW: Load environment variables from .env file

app = Flask(__name__)

# Initialize the advisor
print("Initializing Smart Crop Advisor...")
advisor = SmartCropAdvisor()

# NEW: Setup Gemini API on startup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
api_is_ok, api_startup_message = advisor.setup_gemini_api(GEMINI_API_KEY)
print(api_startup_message) # Print status to the console

# Train the ML model
X, y = advisor.load_and_preprocess_data()
r2, rmse = advisor.train_model(X, y)
print(f"✅ ML Model trained and ready. Performance: R²={r2:.3f}, RMSE={rmse:.3f}")

@app.route('/')
def index():
    """Render the main HTML page."""
    return render_template('index.html')

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

if __name__ == '__main__':
    app.run(debug=True)
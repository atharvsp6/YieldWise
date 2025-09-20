import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import LanguageDetector from 'i18next-browser-languagedetector';

// Translation files
const resources = {
  en: {
    translation: {
      nav: {
        home: "Home",
        yieldPredictor: "Yield Predictor",
        dashboard: "Dashboard", 
        financialCalculator: "Financial Calculator",
        diseaseDetection: "Disease Detection",
        darkMode: "Dark Mode",
        lightMode: "Light Mode"
      },
      home: {
        title: "YieldWise - Smart Agricultural Intelligence",
        subtitle: "AI-Powered Platform for Crop Yield Prediction & Agricultural Analysis",
        description: "Revolutionize your farming with data-driven insights, precise yield predictions, and comprehensive agricultural intelligence.",
        getStarted: "Get Started",
        learnMore: "Learn More",
        features: {
          yieldPrediction: {
            title: "AI Yield Prediction",
            description: "Advanced machine learning models predict crop yields with high accuracy"
          },
          financialAnalysis: {
            title: "Financial Analysis",
            description: "Comprehensive cost analysis and ROI calculations for informed decisions"
          },
          diseaseDetection: {
            title: "Disease Detection", 
            description: "Computer vision technology for early disease identification"
          }
        }
      },
      yieldPredictor: {
        title: "Crop Yield Predictor",
        subtitle: "AI-Powered Crop Yield Forecasting",
        form: {
          crop: "Crop Type",
          season: "Season",
          state: "State/Region",
          area: "Area (hectares)",
          rainfall: "Rainfall (mm)",
          fertilizer: "Fertilizer (kg/ha)",
          pesticide: "Pesticide (kg/ha)",
          predictYield: "Predict Yield",
          loading: "Analyzing data...",
          selectCrop: "Select crop type",
          selectSeason: "Select season",
          selectState: "Select state"
        },
        results: {
          title: "Prediction Results",
          predictedYield: "Predicted Yield",
          confidence: "Confidence Level",
          recommendations: "AI Recommendations",
          error: "Prediction Error"
        }
      },
      dashboard: {
        title: "Agricultural Dashboard",
        subtitle: "Comprehensive Agricultural Insights",
        overview: "Overview",
        recentPredictions: "Recent Predictions",
        weatherInsights: "Weather Insights", 
        marketTrends: "Market Trends"
      },
      financial: {
        title: "Financial Calculator",
        subtitle: "Agricultural Cost Analysis & ROI Calculator",
        costAnalysis: "Cost Analysis",
        roiProjection: "ROI Projection",
        calculate: "Calculate"
      },
      disease: {
        title: "Disease Detection",
        subtitle: "AI-Powered Plant Disease Identification",
        uploadImage: "Upload Plant Image",
        analyzeImage: "Analyze Image",
        dragDrop: "Drag and drop an image or click to browse",
        analyzing: "Analyzing image...",
        results: "Analysis Results"
      },
      common: {
        loading: "Loading...",
        error: "Error",
        tryAgain: "Try Again",
        cancel: "Cancel",
        save: "Save",
        language: "Language"
      }
    }
  },
  es: {
    translation: {
      nav: {
        home: "Inicio",
        yieldPredictor: "Predictor de Rendimiento",
        dashboard: "Panel",
        financialCalculator: "Calculadora Financiera", 
        diseaseDetection: "Detección de Enfermedades",
        darkMode: "Modo Oscuro",
        lightMode: "Modo Claro"
      },
      home: {
        title: "YieldWise - Inteligencia Agrícola Inteligente",
        subtitle: "Plataforma con IA para Predicción de Rendimientos y Análisis Agrícola",
        description: "Revoluciona tu agricultura con conocimientos basados en datos, predicciones precisas de rendimiento e inteligencia agrícola integral.",
        getStarted: "Comenzar",
        learnMore: "Aprender Más",
        features: {
          yieldPrediction: {
            title: "Predicción de Rendimiento con IA",
            description: "Modelos avanzados de aprendizaje automático predicen rendimientos de cultivos con alta precisión"
          },
          financialAnalysis: {
            title: "Análisis Financiero",
            description: "Análisis integral de costos y cálculos de ROI para decisiones informadas"
          },
          diseaseDetection: {
            title: "Detección de Enfermedades",
            description: "Tecnología de visión por computadora para identificación temprana de enfermedades"
          }
        }
      },
      yieldPredictor: {
        title: "Predictor de Rendimiento de Cultivos",
        subtitle: "Pronóstico de Rendimiento de Cultivos con IA",
        form: {
          crop: "Tipo de Cultivo",
          season: "Temporada",
          state: "Estado/Región",
          area: "Área (hectáreas)",
          rainfall: "Precipitación (mm)",
          fertilizer: "Fertilizante (kg/ha)",
          pesticide: "Pesticida (kg/ha)",
          predictYield: "Predecir Rendimiento",
          loading: "Analizando datos...",
          selectCrop: "Seleccionar tipo de cultivo",
          selectSeason: "Seleccionar temporada",
          selectState: "Seleccionar estado"
        },
        results: {
          title: "Resultados de Predicción",
          predictedYield: "Rendimiento Predicho",
          confidence: "Nivel de Confianza",
          recommendations: "Recomendaciones de IA",
          error: "Error de Predicción"
        }
      },
      dashboard: {
        title: "Panel Agrícola",
        subtitle: "Información Agrícola Integral",
        overview: "Resumen",
        recentPredictions: "Predicciones Recientes",
        weatherInsights: "Información Meteorológica",
        marketTrends: "Tendencias del Mercado"
      },
      financial: {
        title: "Calculadora Financiera", 
        subtitle: "Análisis de Costos Agrícolas y Calculadora de ROI",
        costAnalysis: "Análisis de Costos",
        roiProjection: "Proyección de ROI",
        calculate: "Calcular"
      },
      disease: {
        title: "Detección de Enfermedades",
        subtitle: "Identificación de Enfermedades de Plantas con IA",
        uploadImage: "Subir Imagen de Planta",
        analyzeImage: "Analizar Imagen",
        dragDrop: "Arrastra y suelta una imagen o haz clic para navegar",
        analyzing: "Analizando imagen...",
        results: "Resultados del Análisis"
      },
      common: {
        loading: "Cargando...",
        error: "Error",
        tryAgain: "Intentar de Nuevo",
        cancel: "Cancelar",
        save: "Guardar",
        language: "Idioma"
      }
    }
  },
  fr: {
    translation: {
      nav: {
        home: "Accueil",
        yieldPredictor: "Prédicteur de Rendement",
        dashboard: "Tableau de Bord",
        financialCalculator: "Calculatrice Financière",
        diseaseDetection: "Détection de Maladies",
        darkMode: "Mode Sombre",
        lightMode: "Mode Clair"
      },
      home: {
        title: "YieldWise - Intelligence Agricole Intelligente",
        subtitle: "Plateforme IA pour la Prédiction de Rendement et l'Analyse Agricole",
        description: "Révolutionnez votre agriculture avec des insights basés sur les données, des prédictions précises de rendement et une intelligence agricole complète.",
        getStarted: "Commencer",
        learnMore: "En Savoir Plus",
        features: {
          yieldPrediction: {
            title: "Prédiction de Rendement IA",
            description: "Les modèles d'apprentissage automatique avancés prédisent les rendements des cultures avec une grande précision"
          },
          financialAnalysis: {
            title: "Analyse Financière",
            description: "Analyse complète des coûts et calculs de ROI pour des décisions éclairées"
          },
          diseaseDetection: {
            title: "Détection de Maladies",
            description: "Technologie de vision par ordinateur pour l'identification précoce des maladies"
          }
        }
      },
      yieldPredictor: {
        title: "Prédicteur de Rendement des Cultures",
        subtitle: "Prévision de Rendement des Cultures avec IA",
        form: {
          crop: "Type de Culture",
          season: "Saison",
          state: "État/Région",
          area: "Surface (hectares)",
          rainfall: "Précipitations (mm)",
          fertilizer: "Engrais (kg/ha)",
          pesticide: "Pesticide (kg/ha)",
          predictYield: "Prédire le Rendement",
          loading: "Analyse des données...",
          selectCrop: "Sélectionner le type de culture",
          selectSeason: "Sélectionner la saison",
          selectState: "Sélectionner l'état"
        },
        results: {
          title: "Résultats de Prédiction",
          predictedYield: "Rendement Prédit",
          confidence: "Niveau de Confiance",
          recommendations: "Recommandations IA",
          error: "Erreur de Prédiction"
        }
      },
      dashboard: {
        title: "Tableau de Bord Agricole",
        subtitle: "Insights Agricoles Compréhensifs",
        overview: "Aperçu",
        recentPredictions: "Prédictions Récentes",
        weatherInsights: "Insights Météorologiques",
        marketTrends: "Tendances du Marché"
      },
      financial: {
        title: "Calculatrice Financière",
        subtitle: "Analyse des Coûts Agricoles et Calculatrice de ROI",
        costAnalysis: "Analyse des Coûts",
        roiProjection: "Projection ROI",
        calculate: "Calculer"
      },
      disease: {
        title: "Détection de Maladies",
        subtitle: "Identification de Maladies de Plantes avec IA",
        uploadImage: "Télécharger Image de Plante",
        analyzeImage: "Analyser l'Image",
        dragDrop: "Glissez-déposez une image ou cliquez pour naviguer",
        analyzing: "Analyse de l'image...",
        results: "Résultats de l'Analyse"
      },
      common: {
        loading: "Chargement...",
        error: "Erreur",
        tryAgain: "Réessayer",
        cancel: "Annuler",
        save: "Sauvegarder",
        language: "Langue"
      }
    }
  },
  hi: {
    translation: {
      nav: {
        home: "होम",
        yieldPredictor: "उत्पादन भविष्यवक्ता",
        dashboard: "डैशबोर्ड",
        financialCalculator: "वित्तीय कैलकुलेटर",
        diseaseDetection: "बीमारी की जाँच",
        darkMode: "डार्क मोड",
        lightMode: "लाइट मोड"
      },
      home: {
        title: "YieldWise - स्मार्ट कृषि बुद्धिमत्ता",
        subtitle: "फसल उत्पादन भविष्यवाणी और कृषि विश्लेषण के लिए AI-संचालित प्लेटफॉर्म",
        description: "डेटा-संचालित अंतर्दृष्टि, सटीक उत्पादन भविष्यवाणियों और व्यापक कृषि बुद्धिमत्ता के साथ अपनी खेती में क्रांति लाएं।",
        getStarted: "शुरू करें",
        learnMore: "और जानें",
        features: {
          yieldPrediction: {
            title: "AI उत्पादन भविष्यवाणी",
            description: "उन्नत मशीन लर्निंग मॉडल उच्च सटीकता के साथ फसल उत्पादन की भविष्यवाणी करते हैं"
          },
          financialAnalysis: {
            title: "वित्तीय विश्लेषण",
            description: "सूचित निर्णयों के लिए व्यापक लागत विश्लेषण और ROI गणना"
          },
          diseaseDetection: {
            title: "बीमारी की जाँच",
            description: "प्रारंभिक बीमारी की पहचान के लिए कंप्यूटर विजन तकनीक"
          }
        }
      },
      yieldPredictor: {
        title: "फसल उत्पादन भविष्यवक्ता",
        subtitle: "AI-संचालित फसल उत्पादन पूर्वानुमान",
        form: {
          crop: "फसल का प्रकार",
          season: "मौसम",
          state: "राज्य/क्षेत्र",
          area: "क्षेत्रफल (हेक्टेयर)",
          rainfall: "वर्षा (मिमी)",
          fertilizer: "उर्वरक (किग्रा/हेक्टेयर)",
          pesticide: "कीटनाशक (किग्रा/हेक्टेयर)",
          predictYield: "उत्पादन की भविष्यवाणी करें",
          loading: "डेटा का विश्लेषण कर रहे हैं...",
          selectCrop: "फसल का प्रकार चुनें",
          selectSeason: "मौसम चुनें",
          selectState: "राज्य चुनें"
        },
        results: {
          title: "भविष्यवाणी के परिणाम",
          predictedYield: "भविष्यवाणी किया गया उत्पादन",
          confidence: "विश्वास स्तर",
          recommendations: "AI सिफारिशें",
          error: "भविष्यवाणी त्रुटि"
        }
      },
      dashboard: {
        title: "कृषि डैशबोर्ड",
        subtitle: "व्यापक कृषि अंतर्दृष्टि",
        overview: "अवलोकन",
        recentPredictions: "हाल की भविष्यवाणियां",
        weatherInsights: "मौसम की जानकारी",
        marketTrends: "बाजार के रुझान"
      },
      financial: {
        title: "वित्तीय कैलकुलेटर",
        subtitle: "कृषि लागत विश्लेषण और ROI कैलकुलेटर",
        costAnalysis: "लागत विश्लेषण",
        roiProjection: "ROI प्रक्षेपण",
        calculate: "गणना करें"
      },
      disease: {
        title: "बीमारी की जाँच",
        subtitle: "AI-संचालित पौधे की बीमारी की पहचान",
        uploadImage: "पौधे की छवि अपलोड करें",
        analyzeImage: "छवि का विश्लेषण करें",
        dragDrop: "एक छवि को खींचें और छोड़ें या ब्राउज़ करने के लिए क्लिक करें",
        analyzing: "छवि का विश्लेषण कर रहे हैं...",
        results: "विश्लेषण के परिणाम"
      },
      common: {
        loading: "लोड हो रहा है...",
        error: "त्रुटि",
        tryAgain: "पुनः प्रयास करें",
        cancel: "रद्द करें",
        save: "सेव करें",
        language: "भाषा"
      }
    }
  }
};

i18n
  .use(LanguageDetector)
  .use(initReactI18next)
  .init({
    resources,
    fallbackLng: 'en',
    debug: false,
    
    detection: {
      order: ['localStorage', 'navigator', 'htmlTag'],
      caches: ['localStorage'],
    },

    interpolation: {
      escapeValue: false,
    }
  });

export default i18n;
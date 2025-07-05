from django.apps import AppConfig
import os
import joblib
import logging
import sys
from django.conf import settings

# Define tokenizer function at the top level
def custom_tokenizer(text):
    return text.split('-') if text else []

# Make it available in the main module
sys.modules['__main__'].custom_tokenizer = custom_tokenizer

logger = logging.getLogger(__name__)

class HomeConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'Home'
    
    gpa_model = None
    tfidf = None
    model_features = None
    
    def ready(self):
        # Define paths to model files
        base_dir = settings.BASE_DIR
        model_path = os.path.join(base_dir, 'Home', 'final_gpa_predictor_rf.pkl')
        tfidf_path = os.path.join(base_dir, 'Home', 'tfidf_vectorizer.pkl')
        features_path = os.path.join(base_dir, 'Home', 'model_features.pkl')
        
        # Ensure tokenizer is registered in main
        sys.modules['__main__'].custom_tokenizer = custom_tokenizer
        
        # Load GPA model
        try:
            if os.path.exists(model_path):
                self.gpa_model = joblib.load(model_path)
                logger.info(f"✅ GPA model loaded successfully. Type: {type(self.gpa_model)}")
            else:
                logger.error(f"⚠️ Model file not found: {model_path}")
        except Exception as e:
            logger.error(f"❌ Error loading GPA model: {str(e)}")
            self.gpa_model = None
        
        # Load TF-IDF vectorizer
        try:
            if os.path.exists(tfidf_path):
                self.tfidf = joblib.load(tfidf_path)
                logger.info(f"✅ TF-IDF vectorizer loaded successfully. Type: {type(self.tfidf)}")
            else:
                logger.error(f"⚠️ TF-IDF file not found: {tfidf_path}")
        except Exception as e:
            logger.error(f"❌ Error loading TF-IDF vectorizer: {str(e)}")
            self.tfidf = None
        
        # Load model features
        try:
            if os.path.exists(features_path):
                self.model_features = joblib.load(features_path)
                logger.info("✅ Model features loaded successfully")
            else:
                logger.error(f"⚠️ Features file not found: {features_path}")
        except Exception as e:
            logger.error(f"❌ Error loading model features: {str(e)}")
            self.model_features = None
        
        # Final validation
        if self.gpa_model and self.tfidf and self.model_features:
            logger.info("✅ All GPA models loaded successfully")
        else:
            logger.error("❌ Some models failed to load")
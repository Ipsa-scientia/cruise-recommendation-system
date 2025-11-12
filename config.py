import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev_secret_key')
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    DEBUG = os.getenv('FLASK_DEBUG', '0').lower() in ('1', 'true', 't')
    CSV_FILE_PATH = os.path.join(os.path.dirname(__file__), 'data', 'cruises_data.csv')

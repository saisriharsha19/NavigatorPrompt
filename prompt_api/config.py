# config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # LLM Config
    API_KEY = os.getenv("UFL_AI_API_KEY")
    BASE_URL = os.getenv("UFL_AI_BASE_URL")
    MODEL_NAME = os.getenv("UFL_AI_MODEL")

    # Database
    DATABASE_URL = os.getenv("DATABASE_URL")

    # Celery & Redis Sentinel
    CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL")
    CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND")
    REDIS_SERVICE_NAME = os.getenv("REDIS_SERVICE_NAME")
    
    # Admin Configuration
    ADMIN_KEY = os.getenv("ADMIN_KEY")
    
    # Security
    ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-here")
    JWT_ALGORITHM = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))  # 24 hours
    
    # SAML Configuration
    SAML_SP_ENTITY_ID = os.getenv("SAML_SP_ENTITY_ID", "https://your-app.ufl.edu")
    SAML_SP_ACS_URL = os.getenv("SAML_SP_ACS_URL", "https://your-app.ufl.edu/auth/saml/callback")
    SAML_SP_SLS_URL = os.getenv("SAML_SP_SLS_URL", "https://your-app.ufl.edu/auth/saml/logout")
    SAML_IDP_ENTITY_ID = os.getenv("SAML_IDP_ENTITY_ID", "https://login.ufl.edu/idp/shibboleth")
    SAML_IDP_SSO_URL = os.getenv("SAML_IDP_SSO_URL", "https://login.ufl.edu/idp/profile/SAML2/Redirect/SSO")
    SAML_IDP_SLO_URL = os.getenv("SAML_IDP_SLO_URL", "https://login.ufl.edu/idp/profile/SAML2/Redirect/SLO")
    SAML_IDP_X509_CERT = os.getenv("SAML_IDP_X509_CERT", "")  # Base64 encoded certificate
    SAML_SP_X509_CERT = os.getenv("SAML_SP_X509_CERT", "")
    SAML_SP_PRIVATE_KEY = os.getenv("SAML_SP_PRIVATE_KEY", "")
    
    # Caching
    ENABLE_CACHING = os.getenv("ENABLE_CACHING", "true").lower() == "true"
    
    # Server Config
    PORT = int(os.getenv("PORT", 5000))
    HOST = os.getenv("HOST", "0.0.0.0")
    
    # University Domain Validation
    ALLOWED_EMAIL_DOMAINS = os.getenv("ALLOWED_EMAIL_DOMAINS", "ufl.edu,ad.ufl.edu").split(",")

settings = Settings()
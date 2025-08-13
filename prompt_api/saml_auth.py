# saml_auth.py
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import json
import base64
from urllib.parse import urlparse

from jose import JWTError, jwt
from passlib.context import CryptContext
from onelogin.saml2.auth import OneLogin_Saml2_Auth
from onelogin.saml2.settings import OneLogin_Saml2_Settings
from onelogin.saml2.utils import OneLogin_Saml2_Utils

from config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_saml_settings() -> Dict[str, Any]:
    """Get SAML configuration settings"""
    return {
        "sp": {
            "entityId": settings.SAML_SP_ENTITY_ID,
            "assertionConsumerService": {
                "url": settings.SAML_SP_ACS_URL,
                "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST"
            },
            "singleLogoutService": {
                "url": settings.SAML_SP_SLS_URL,
                "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect"
            },
            "NameIDFormat": "urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress",
            "x509cert": settings.SAML_SP_X509_CERT,
            "privateKey": settings.SAML_SP_PRIVATE_KEY
        },
        "idp": {
            "entityId": settings.SAML_IDP_ENTITY_ID,
            "singleSignOnService": {
                "url": settings.SAML_IDP_SSO_URL,
                "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect"
            },
            "singleLogoutService": {
                "url": settings.SAML_IDP_SLO_URL,
                "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect"
            },
            "x509cert": settings.SAML_IDP_X509_CERT
        }
    }

def init_saml_auth(req: Dict[str, Any]) -> OneLogin_Saml2_Auth:
    """Initialize SAML authentication"""
    saml_settings = get_saml_settings()
    auth = OneLogin_Saml2_Auth(req, saml_settings)
    return auth

def prepare_fastapi_request(request) -> Dict[str, Any]:
    """Prepare FastAPI request for SAML"""
    url_data = urlparse(str(request.url))
    return {
        'https': 'on' if request.url.scheme == 'https' else 'off',
        'http_host': request.headers.get('host', url_data.netloc),
        'server_port': url_data.port or (443 if url_data.scheme == 'https' else 80),
        'script_name': url_data.path,
        'get_data': dict(request.query_params),
        'post_data': dict(request.form) if hasattr(request, 'form') else {}
    }

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify JWT token"""
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        return payload
    except JWTError:
        return None

def extract_user_info_from_saml(auth: OneLogin_Saml2_Auth) -> Dict[str, Any]:
    """Extract user information from SAML response"""
    attributes = auth.get_attributes()
    
    # Map SAML attributes to user fields (adjust based on your IDP's attribute mapping)
    user_info = {
        "email": auth.get_nameid(),  # Usually email for UFL
        "first_name": attributes.get('urn:oid:2.5.4.42', [''])[0] or attributes.get('givenName', [''])[0],  # Given name
        "last_name": attributes.get('urn:oid:2.5.4.4', [''])[0] or attributes.get('sn', [''])[0],  # Surname
        "username": attributes.get('urn:oid:0.9.2342.19200300.100.1.1', [''])[0] or attributes.get('uid', [''])[0],  # Username
        "student_id": attributes.get('urn:oid:2.16.840.1.113730.3.1.3', [''])[0] or '',  # Employee/Student ID
        "affiliation": attributes.get('urn:oid:1.3.6.1.4.1.5923.1.1.1.1', [''])[0] or '',  # eduPersonAffiliation
    }
    
    # Fallback logic for username and full name
    if not user_info["username"] and user_info["email"]:
        user_info["username"] = user_info["email"].split('@')[0]
    
    if not user_info["first_name"] and not user_info["last_name"]:
        # Try common name attribute
        common_name = attributes.get('urn:oid:2.5.4.3', [''])[0] or attributes.get('cn', [''])[0]
        if common_name:
            name_parts = common_name.split(' ', 1)
            user_info["first_name"] = name_parts[0]
            user_info["last_name"] = name_parts[1] if len(name_parts) > 1 else ''
    
    return user_info

def is_valid_university_email(email: str) -> bool:
    """Check if email belongs to allowed university domains"""
    if not email:
        return False
    
    domain = email.split('@')[-1].lower()
    return domain in [d.lower() for d in settings.ALLOWED_EMAIL_DOMAINS]

def determine_user_role(user_info: Dict[str, Any]) -> Dict[str, bool]:
    """Determine user role based on SAML attributes"""
    affiliation = user_info.get("affiliation", "").lower()
    email = user_info.get("email", "").lower()
    
    # Default roles
    roles = {
        "is_student": False,
        "is_faculty": False,
        "is_staff": False,
        "is_admin": False
    }
    
    # Determine roles based on affiliation
    if "student" in affiliation:
        roles["is_student"] = True
    if "faculty" in affiliation or "teacher" in affiliation:
        roles["is_faculty"] = True
    if "staff" in affiliation or "employee" in affiliation:
        roles["is_staff"] = True
    
    # Admin determination (you can customize this logic)
    admin_emails = ["admin@ufl.edu"]  # Add specific admin emails
    if email in admin_emails or "admin" in email:
        roles["is_admin"] = True
    
    return roles
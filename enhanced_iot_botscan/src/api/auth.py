"""
Authentication and Authorization Module
Implements SRS-NF-006, SRS-NF-007 requirements

Author: Kotiwale Sumesh Singh (160124862043)
"""

import jwt
import bcrypt
import secrets
import pyotp
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import logging
from enum import Enum
import json
import os

logger = logging.getLogger(__name__)

class UserRole(Enum):
    """User roles for RBAC (SRS-NF-007)"""
    ADMIN = "admin"
    ANALYST = "analyst"
    OPERATOR = "operator"
    VIEWER = "viewer"

class Permission(Enum):
    """System permissions"""
    VIEW_DASHBOARD = "view_dashboard"
    RUN_DETECTION = "run_detection"
    VIEW_ANALYTICS = "view_analytics"
    MANAGE_CONFIG = "manage_config"
    MANAGE_USERS = "manage_users"
    TRAIN_MODEL = "train_model"
    EXPORT_DATA = "export_data"
    VIEW_LOGS = "view_logs"

# Role-Permission mapping (SRS-NF-007)
ROLE_PERMISSIONS = {
    UserRole.ADMIN: [p for p in Permission],  # All permissions
    UserRole.ANALYST: [
        Permission.VIEW_DASHBOARD,
        Permission.RUN_DETECTION,
        Permission.VIEW_ANALYTICS,
        Permission.EXPORT_DATA,
        Permission.VIEW_LOGS
    ],
    UserRole.OPERATOR: [
        Permission.VIEW_DASHBOARD,
        Permission.RUN_DETECTION,
        Permission.VIEW_ANALYTICS
    ],
    UserRole.VIEWER: [
        Permission.VIEW_DASHBOARD,
        Permission.VIEW_ANALYTICS
    ]
}

class User:
    """User model"""
    
    def __init__(self, username: str, password_hash: str, role: UserRole, 
                 mfa_secret: Optional[str] = None, email: Optional[str] = None):
        self.username = username
        self.password_hash = password_hash
        self.role = role
        self.mfa_secret = mfa_secret
        self.email = email
        self.created_at = datetime.now()
        self.last_login = None
        self.is_active = True
        self.failed_login_attempts = 0
        self.locked_until = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary"""
        return {
            'username': self.username,
            'role': self.role.value,
            'email': self.email,
            'created_at': self.created_at.isoformat(),
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'is_active': self.is_active,
            'mfa_enabled': self.mfa_secret is not None
        }

class AuthenticationManager:
    """
    Authentication and Authorization Manager
    Implements SRS-NF-006, SRS-NF-007
    """
    
    def __init__(self, secret_key: str = None, token_expiry: int = 3600):
        self.secret_key = secret_key or os.getenv('SECRET_KEY', secrets.token_hex(32))
        self.token_expiry = token_expiry
        
        # In-memory user storage (replace with database in production)
        self.users: Dict[str, User] = {}
        
        # Session tracking
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Audit log
        self.audit_log: List[Dict[str, Any]] = []
        
        # Create default admin user
        self._create_default_admin()
        
        logger.info("Authentication manager initialized")
    
    def _create_default_admin(self):
        """Create default admin user"""
        admin_password = os.getenv('ADMIN_PASSWORD', 'admin123')
        self.create_user('admin', admin_password, UserRole.ADMIN, 'admin@example.com')
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt (SRS-NF-008)"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        try:
            return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False
    
    def create_user(self, username: str, password: str, role: UserRole, 
                   email: Optional[str] = None) -> bool:
        """
        Create new user
        Implements SRS-NF-007
        """
        try:
            if username in self.users:
                logger.warning(f"User {username} already exists")
                return False
            
            password_hash = self.hash_password(password)
            user = User(username, password_hash, role, email=email)
            self.users[username] = user
            
            self._log_audit_event('user_created', username, {'role': role.value})
            logger.info(f"User {username} created with role {role.value}")
            
            return True
            
        except Exception as e:
            logger.error(f"User creation error: {e}")
            return False
    
    def authenticate(self, username: str, password: str, 
                    mfa_code: Optional[str] = None) -> Optional[str]:
        """
        Authenticate user and return JWT token
        Implements SRS-NF-006
        """
        try:
            user = self.users.get(username)
            
            if not user or not user.is_active:
                self._log_audit_event('login_failed', username, {'reason': 'invalid_user'})
                return None
            
            # Check account lockout
            if user.locked_until and datetime.now() < user.locked_until:
                self._log_audit_event('login_failed', username, {'reason': 'account_locked'})
                return None
            
            # Verify password
            if not self.verify_password(password, user.password_hash):
                user.failed_login_attempts += 1
                
                # Lock account after 5 failed attempts
                if user.failed_login_attempts >= 5:
                    user.locked_until = datetime.now() + timedelta(minutes=30)
                    self._log_audit_event('account_locked', username)
                
                self._log_audit_event('login_failed', username, {'reason': 'invalid_password'})
                return None
            
            # Verify MFA if enabled (SRS-NF-006)
            if user.mfa_secret:
                if not mfa_code or not self.verify_mfa(user.mfa_secret, mfa_code):
                    self._log_audit_event('login_failed', username, {'reason': 'invalid_mfa'})
                    return None
            
            # Reset failed attempts
            user.failed_login_attempts = 0
            user.last_login = datetime.now()
            
            # Generate JWT token
            token = self.generate_token(user)
            
            # Track session
            self.active_sessions[token] = {
                'username': username,
                'role': user.role.value,
                'created_at': datetime.now(),
                'expires_at': datetime.now() + timedelta(seconds=self.token_expiry)
            }
            
            self._log_audit_event('login_success', username)
            logger.info(f"User {username} authenticated successfully")
            
            return token
            
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return None
    
    def generate_token(self, user: User) -> str:
        """Generate JWT token"""
        payload = {
            'user': user.username,
            'role': user.role.value,
            'exp': datetime.utcnow() + timedelta(seconds=self.token_expiry),
            'iat': datetime.utcnow()
        }
        
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            
            # Check if session is still active
            if token not in self.active_sessions:
                return None
            
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid token")
            return None
    
    def logout(self, token: str) -> bool:
        """Logout user and invalidate token"""
        try:
            if token in self.active_sessions:
                username = self.active_sessions[token]['username']
                del self.active_sessions[token]
                self._log_audit_event('logout', username)
                logger.info(f"User {username} logged out")
                return True
            return False
        except Exception as e:
            logger.error(f"Logout error: {e}")
            return False
    
    def setup_mfa(self, username: str) -> Optional[Dict[str, str]]:
        """
        Setup MFA for user (SRS-NF-006)
        Returns QR code data and secret
        """
        try:
            user = self.users.get(username)
            if not user:
                return None
            
            # Generate MFA secret
            secret = pyotp.random_base32()
            user.mfa_secret = secret
            
            # Generate provisioning URI for QR code
            totp = pyotp.TOTP(secret)
            provisioning_uri = totp.provisioning_uri(
                name=username,
                issuer_name="Enhanced IoT BotScan"
            )
            
            self._log_audit_event('mfa_enabled', username)
            
            return {
                'secret': secret,
                'provisioning_uri': provisioning_uri
            }
            
        except Exception as e:
            logger.error(f"MFA setup error: {e}")
            return None
    
    def verify_mfa(self, secret: str, code: str) -> bool:
        """Verify MFA code"""
        try:
            totp = pyotp.TOTP(secret)
            return totp.verify(code, valid_window=1)
        except Exception as e:
            logger.error(f"MFA verification error: {e}")
            return False
    
    def check_permission(self, username: str, permission: Permission) -> bool:
        """
        Check if user has specific permission
        Implements SRS-NF-007 RBAC
        """
        user = self.users.get(username)
        if not user or not user.is_active:
            return False
        
        return permission in ROLE_PERMISSIONS.get(user.role, [])
    
    def has_any_permission(self, username: str, permissions: List[Permission]) -> bool:
        """Check if user has any of the specified permissions"""
        return any(self.check_permission(username, p) for p in permissions)
    
    def has_all_permissions(self, username: str, permissions: List[Permission]) -> bool:
        """Check if user has all specified permissions"""
        return all(self.check_permission(username, p) for p in permissions)
    
    def _log_audit_event(self, event_type: str, username: str, 
                        details: Optional[Dict[str, Any]] = None):
        """
        Log audit event
        Implements SRS-NF-023
        """
        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'username': username,
            'details': details or {}
        }
        
        self.audit_log.append(audit_entry)
        logger.info(f"Audit: {event_type} - {username}")
    
    def get_audit_log(self, username: Optional[str] = None, 
                     start_date: Optional[datetime] = None,
                     end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Retrieve audit log
        Implements SRS-NF-023
        """
        filtered_log = self.audit_log
        
        if username:
            filtered_log = [e for e in filtered_log if e['username'] == username]
        
        if start_date:
            filtered_log = [e for e in filtered_log 
                          if datetime.fromisoformat(e['timestamp']) >= start_date]
        
        if end_date:
            filtered_log = [e for e in filtered_log 
                          if datetime.fromisoformat(e['timestamp']) <= end_date]
        
        return filtered_log
    
    def get_user_info(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user information"""
        user = self.users.get(username)
        return user.to_dict() if user else None
    
    def update_user_role(self, username: str, new_role: UserRole, 
                        admin_user: str) -> bool:
        """Update user role (requires admin permission)"""
        try:
            if not self.check_permission(admin_user, Permission.MANAGE_USERS):
                logger.warning(f"Unauthorized role update attempt by {admin_user}")
                return False
            
            user = self.users.get(username)
            if not user:
                return False
            
            old_role = user.role
            user.role = new_role
            
            self._log_audit_event('role_updated', username, {
                'old_role': old_role.value,
                'new_role': new_role.value,
                'updated_by': admin_user
            })
            
            logger.info(f"User {username} role updated to {new_role.value} by {admin_user}")
            return True
            
        except Exception as e:
            logger.error(f"Role update error: {e}")
            return False
    
    def deactivate_user(self, username: str, admin_user: str) -> bool:
        """Deactivate user account"""
        try:
            if not self.check_permission(admin_user, Permission.MANAGE_USERS):
                return False
            
            user = self.users.get(username)
            if not user:
                return False
            
            user.is_active = False
            self._log_audit_event('user_deactivated', username, {'by': admin_user})
            
            # Invalidate all active sessions
            sessions_to_remove = [token for token, session in self.active_sessions.items()
                                if session['username'] == username]
            for token in sessions_to_remove:
                del self.active_sessions[token]
            
            logger.info(f"User {username} deactivated by {admin_user}")
            return True
            
        except Exception as e:
            logger.error(f"User deactivation error: {e}")
            return False

# Example usage
if __name__ == '__main__':
    auth_manager = AuthenticationManager()
    
    # Create users
    auth_manager.create_user('analyst1', 'password123', UserRole.ANALYST)
    auth_manager.create_user('operator1', 'password123', UserRole.OPERATOR)
    
    # Authenticate
    token = auth_manager.authenticate('admin', 'admin123')
    if token:
        print(f"Authentication successful: {token}")
        
        # Check permissions
        print(f"Can manage config: {auth_manager.check_permission('admin', Permission.MANAGE_CONFIG)}")
        print(f"Can train model: {auth_manager.check_permission('analyst1', Permission.TRAIN_MODEL)}")

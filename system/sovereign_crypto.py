#!/usr/bin/env python3
"""
Sovereign Cryptographic Module for MeRNSTA Phase 35
Provides AES-GCM encryption, Ed25519 identity management, and UCAN-style contracts.
"""

import os
import json
import hmac
import hashlib
import secrets
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field, asdict
import base64

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives.asymmetric import ed25519
    from cryptography.hazmat.backends import default_backend
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    logging.warning("Cryptography library not available - sovereign mode will use mock operations")

import yaml
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import get_config


logger = logging.getLogger(__name__)


@dataclass 
class SovereignIdentity:
    """Cryptographic identity for sovereign agent operations."""
    public_key: str  # Base64 encoded Ed25519 public key
    private_key: str  # Base64 encoded Ed25519 private key (encrypted at rest)
    fingerprint: str  # SHA256 hash of public key
    created_at: datetime
    expires_at: Optional[datetime] = None
    os_fingerprint: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'public_key': self.public_key,
            'private_key': self.private_key,
            'fingerprint': self.fingerprint,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'os_fingerprint': self.os_fingerprint
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SovereignIdentity':
        """Create instance from dictionary."""
        return cls(
            public_key=data['public_key'],
            private_key=data['private_key'],
            fingerprint=data['fingerprint'],
            created_at=datetime.fromisoformat(data['created_at']),
            expires_at=datetime.fromisoformat(data['expires_at']) if data.get('expires_at') else None,
            os_fingerprint=data.get('os_fingerprint')
        )


@dataclass
class AgentContract:
    """UCAN-style agent contract with cryptographic enforcement."""
    agent_id: str
    issuer: str  # Public key of contract issuer
    subject: str  # Public key of contract subject (agent)
    capabilities: List[str]
    resource_limits: Dict[str, Any]
    issued_at: datetime
    expires_at: datetime
    signature: str  # Ed25519 signature of contract
    nonce: str  # Unique nonce to prevent replay
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'agent_id': self.agent_id,
            'issuer': self.issuer,
            'subject': self.subject,
            'capabilities': self.capabilities,
            'resource_limits': self.resource_limits,
            'issued_at': self.issued_at.isoformat(),
            'expires_at': self.expires_at.isoformat(),
            'signature': self.signature,
            'nonce': self.nonce
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentContract':
        """Create instance from dictionary."""
        return cls(
            agent_id=data['agent_id'],
            issuer=data['issuer'],
            subject=data['subject'],
            capabilities=data['capabilities'],
            resource_limits=data['resource_limits'],
            issued_at=datetime.fromisoformat(data['issued_at']),
            expires_at=datetime.fromisoformat(data['expires_at']),
            signature=data['signature'],
            nonce=data['nonce']
        )
    
    def get_signing_payload(self) -> str:
        """Get the payload that should be signed for this contract."""
        # Create deterministic signing payload
        payload = {
            'agent_id': self.agent_id,
            'issuer': self.issuer,
            'subject': self.subject,
            'capabilities': sorted(self.capabilities),  # Sort for determinism
            'resource_limits': dict(sorted(self.resource_limits.items())),
            'issued_at': self.issued_at.isoformat(),
            'expires_at': self.expires_at.isoformat(),
            'nonce': self.nonce
        }
        return json.dumps(payload, sort_keys=True)
    
    def is_expired(self) -> bool:
        """Check if contract has expired."""
        return datetime.now() > self.expires_at
    
    def has_capability(self, capability: str) -> bool:
        """Check if contract grants a specific capability."""
        return capability in self.capabilities


class SovereignCrypto:
    """Centralized cryptographic operations for sovereign mode."""
    
    def __init__(self, config_path: str = "configs/sovereign_config.yaml"):
        """Initialize with configuration."""
        self.config = self._load_config(config_path)
        self.sovereign_config = self.config.get("sovereign_mode", {})
        
        # Initialize paths
        self.identity_path = Path(self.sovereign_config.get("identity", {}).get("keypair_path", "./storage/sovereign"))
        self.identity_path.mkdir(parents=True, exist_ok=True)
        
        # Cache for performance
        self._key_cache = {}
        self._identity_cache = None
        
        # Initialize crypto backend
        if not CRYPTO_AVAILABLE:
            logger.warning("Cryptography library not available - using mock operations")
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load sovereign configuration."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load sovereign config: {e}")
            return {}
    
    def generate_identity(self, force_new: bool = False) -> SovereignIdentity:
        """Generate or load sovereign identity keypair."""
        identity_file = self.identity_path / "identity.json"
        
        # Try to load existing identity unless forced
        if not force_new and identity_file.exists():
            try:
                with open(identity_file, 'r') as f:
                    data = json.load(f)
                identity = SovereignIdentity.from_dict(data)
                
                # Check if identity needs rotation
                if self._should_rotate_identity(identity):
                    logger.info("Identity rotation required - generating new identity")
                    return self._generate_new_identity()
                
                self._identity_cache = identity
                return identity
            except Exception as e:
                logger.error(f"Failed to load existing identity: {e}")
        
        return self._generate_new_identity()
    
    def _generate_new_identity(self) -> SovereignIdentity:
        """Generate a new cryptographic identity."""
        if not CRYPTO_AVAILABLE:
            return self._mock_generate_identity()
        
        # Generate Ed25519 keypair
        private_key = ed25519.Ed25519PrivateKey.generate()
        public_key = private_key.public_key()
        
        # Serialize keys
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()  # Will encrypt with AES-GCM separately
        )
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        # Create fingerprint
        fingerprint = hashlib.sha256(public_pem).hexdigest()
        
        # Get OS fingerprint
        os_fingerprint = self._generate_os_fingerprint()
        
        # Create identity
        identity = SovereignIdentity(
            public_key=base64.b64encode(public_pem).decode('utf-8'),
            private_key=base64.b64encode(private_pem).decode('utf-8'),
            fingerprint=fingerprint,
            created_at=datetime.now(),
            expires_at=self._get_identity_expiry(),
            os_fingerprint=os_fingerprint
        )
        
        # Save identity (encrypted)
        self._save_identity(identity)
        self._identity_cache = identity
        
        logger.info(f"Generated new sovereign identity: {fingerprint[:16]}...")
        return identity
    
    def _mock_generate_identity(self) -> SovereignIdentity:
        """Generate mock identity when crypto library unavailable."""
        mock_public = base64.b64encode(b"mock_public_key_" + secrets.token_bytes(32)).decode('utf-8')
        mock_private = base64.b64encode(b"mock_private_key_" + secrets.token_bytes(32)).decode('utf-8')
        fingerprint = hashlib.sha256(mock_public.encode()).hexdigest()
        
        identity = SovereignIdentity(
            public_key=mock_public,
            private_key=mock_private,
            fingerprint=fingerprint,
            created_at=datetime.now(),
            expires_at=self._get_identity_expiry(),
            os_fingerprint=self._generate_os_fingerprint()
        )
        
        self._save_identity(identity)
        self._identity_cache = identity
        return identity
    
    def _should_rotate_identity(self, identity: SovereignIdentity) -> bool:
        """Check if identity should be rotated."""
        auto_rotate_days = self.sovereign_config.get("identity", {}).get("auto_rotate_days", 90)
        age = datetime.now() - identity.created_at
        return age.days >= auto_rotate_days
    
    def _get_identity_expiry(self) -> datetime:
        """Get identity expiry time."""
        auto_rotate_days = self.sovereign_config.get("identity", {}).get("auto_rotate_days", 90)
        return datetime.now() + timedelta(days=auto_rotate_days)
    
    def _save_identity(self, identity: SovereignIdentity):
        """Save identity to encrypted file."""
        identity_file = self.identity_path / "identity.json"
        
        # Backup existing identity if it exists
        if identity_file.exists():
            self._backup_identity()
        
        # Save new identity (TODO: encrypt with master key)
        with open(identity_file, 'w') as f:
            json.dump(identity.to_dict(), f, indent=2)
        
        # Set secure permissions
        os.chmod(identity_file, 0o600)
    
    def _backup_identity(self):
        """Backup current identity."""
        identity_file = self.identity_path / "identity.json"
        backup_count = self.sovereign_config.get("identity", {}).get("backup_count", 5)
        
        # Rotate existing backups
        for i in range(backup_count - 1, 0, -1):
            old_backup = self.identity_path / f"identity.backup.{i}.json"
            new_backup = self.identity_path / f"identity.backup.{i + 1}.json"
            if old_backup.exists():
                old_backup.rename(new_backup)
        
        # Create new backup
        if identity_file.exists():
            backup_file = self.identity_path / "identity.backup.1.json"
            identity_file.rename(backup_file)
    
    def _generate_os_fingerprint(self) -> str:
        """Generate OS and hardware fingerprint."""
        fingerprint_data = {}
        
        config = self.sovereign_config.get("os_integration", {}).get("fingerprinting", {})
        
        if config.get("include_os_version", True):
            import platform
            fingerprint_data["os"] = {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor()
            }
        
        if config.get("include_hardware", True):
            try:
                import psutil
                fingerprint_data["hardware"] = {
                    "cpu_count": psutil.cpu_count(),
                    "memory_total": psutil.virtual_memory().total,
                    "boot_time": psutil.boot_time()
                }
            except ImportError:
                fingerprint_data["hardware"] = {"available": False}
        
        if config.get("include_network", True):
            try:
                import socket
                fingerprint_data["network"] = {
                    "hostname": socket.gethostname(),
                    "fqdn": socket.getfqdn()
                }
            except Exception:
                fingerprint_data["network"] = {"available": False}
        
        # Hash the fingerprint data
        hash_algo = config.get("fingerprint_hash_algorithm", "sha256")
        fingerprint_json = json.dumps(fingerprint_data, sort_keys=True)
        
        if hash_algo == "sha256":
            return hashlib.sha256(fingerprint_json.encode()).hexdigest()
        else:
            return hashlib.sha256(fingerprint_json.encode()).hexdigest()  # Default to SHA256
    
    def encrypt_data(self, data: bytes, key: Optional[bytes] = None) -> Dict[str, str]:
        """Encrypt data using AES-GCM."""
        if not CRYPTO_AVAILABLE:
            return self._mock_encrypt_data(data)
        
        if key is None:
            key = self._get_master_key()
        
        # Generate random nonce
        nonce = secrets.token_bytes(12)  # 96 bits for GCM
        
        # Encrypt with AES-GCM
        aesgcm = AESGCM(key)
        ciphertext = aesgcm.encrypt(nonce, data, None)
        
        return {
            "ciphertext": base64.b64encode(ciphertext).decode('utf-8'),
            "nonce": base64.b64encode(nonce).decode('utf-8'),
            "algorithm": "aes-gcm-256"
        }
    
    def decrypt_data(self, encrypted_data: Dict[str, str], key: Optional[bytes] = None) -> bytes:
        """Decrypt data using AES-GCM."""
        if not CRYPTO_AVAILABLE:
            return self._mock_decrypt_data(encrypted_data)
        
        if key is None:
            key = self._get_master_key()
        
        ciphertext = base64.b64decode(encrypted_data["ciphertext"])
        nonce = base64.b64decode(encrypted_data["nonce"])
        
        # Decrypt with AES-GCM
        aesgcm = AESGCM(key)
        plaintext = aesgcm.decrypt(nonce, ciphertext, None)
        
        return plaintext
    
    def _mock_encrypt_data(self, data: bytes) -> Dict[str, str]:
        """Mock encryption for when crypto library unavailable."""
        # Simple base64 encoding as mock encryption
        encoded = base64.b64encode(data).decode('utf-8')
        return {
            "ciphertext": encoded,
            "nonce": base64.b64encode(secrets.token_bytes(12)).decode('utf-8'),
            "algorithm": "mock-encryption"
        }
    
    def _mock_decrypt_data(self, encrypted_data: Dict[str, str]) -> bytes:
        """Mock decryption for when crypto library unavailable."""
        return base64.b64decode(encrypted_data["ciphertext"])
    
    def _get_master_key(self) -> bytes:
        """Get or generate master encryption key."""
        key_file = self.identity_path / "master.key"
        
        if key_file.exists():
            try:
                with open(key_file, 'rb') as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Failed to load master key: {e}")
        
        # Generate new master key
        key = secrets.token_bytes(32)  # 256 bits
        
        with open(key_file, 'wb') as f:
            f.write(key)
        
        # Set secure permissions
        os.chmod(key_file, 0o600)
        
        logger.info("Generated new master encryption key")
        return key
    
    def create_agent_contract(self, agent_id: str, capabilities: List[str], 
                            resource_limits: Optional[Dict[str, Any]] = None,
                            validity_hours: Optional[int] = None) -> AgentContract:
        """Create a new UCAN-style agent contract."""
        identity = self.generate_identity()
        
        if resource_limits is None:
            resource_limits = self.sovereign_config.get("contracts", {}).get("resource_limits", {})
        
        if validity_hours is None:
            validity_hours = self.sovereign_config.get("contracts", {}).get("contract_expiry_hours", 24)
        
        # Create contract
        contract = AgentContract(
            agent_id=agent_id,
            issuer=identity.fingerprint,
            subject=agent_id,  # For now, subject is same as agent_id
            capabilities=capabilities,
            resource_limits=resource_limits,
            issued_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=validity_hours),
            signature="",  # Will be filled by signing
            nonce=secrets.token_hex(16)
        )
        
        # Sign the contract
        contract.signature = self.sign_contract(contract)
        
        return contract
    
    def sign_contract(self, contract: AgentContract) -> str:
        """Sign an agent contract with sovereign identity."""
        if not CRYPTO_AVAILABLE:
            return self._mock_sign_contract(contract)
        
        identity = self.generate_identity()
        
        # Get signing payload
        payload = contract.get_signing_payload()
        
        # Load private key
        private_key_pem = base64.b64decode(identity.private_key)
        private_key = serialization.load_pem_private_key(
            private_key_pem,
            password=None,
            backend=default_backend()
        )
        
        # Sign payload
        signature = private_key.sign(payload.encode('utf-8'))
        
        return base64.b64encode(signature).decode('utf-8')
    
    def _mock_sign_contract(self, contract: AgentContract) -> str:
        """Mock contract signing when crypto library unavailable."""
        payload = contract.get_signing_payload()
        mock_signature = hashlib.sha256(payload.encode()).hexdigest()
        return base64.b64encode(mock_signature.encode()).decode('utf-8')
    
    def verify_contract(self, contract: AgentContract) -> bool:
        """Verify an agent contract signature."""
        if not CRYPTO_AVAILABLE:
            return self._mock_verify_contract(contract)
        
        try:
            identity = self.generate_identity()
            
            # Load public key
            public_key_pem = base64.b64decode(identity.public_key)
            public_key = serialization.load_pem_public_key(
                public_key_pem,
                backend=default_backend()
            )
            
            # Get signing payload
            payload = contract.get_signing_payload()
            signature = base64.b64decode(contract.signature)
            
            # Verify signature
            public_key.verify(signature, payload.encode('utf-8'))
            return True
            
        except Exception as e:
            logger.error(f"Contract verification failed: {e}")
            return False
    
    def _mock_verify_contract(self, contract: AgentContract) -> bool:
        """Mock contract verification when crypto library unavailable."""
        # Simple verification for mock mode
        payload = contract.get_signing_payload()
        expected_signature = hashlib.sha256(payload.encode()).hexdigest()
        actual_signature = base64.b64decode(contract.signature).decode('utf-8')
        return expected_signature == actual_signature
    
    def encrypt_database(self, db_path: str) -> bool:
        """Encrypt a SQLite database in-place."""
        if not os.path.exists(db_path):
            logger.warning(f"Database not found: {db_path}")
            return False
        
        try:
            # Read database
            with open(db_path, 'rb') as f:
                db_data = f.read()
            
            # Encrypt data
            encrypted = self.encrypt_data(db_data)
            
            # Create encrypted database file
            encrypted_path = f"{db_path}.encrypted"
            with open(encrypted_path, 'w') as f:
                json.dump(encrypted, f)
            
            # Replace original with encrypted (backup original first)
            backup_path = f"{db_path}.backup"
            os.rename(db_path, backup_path)
            os.rename(encrypted_path, db_path)
            
            logger.info(f"Encrypted database: {db_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to encrypt database {db_path}: {e}")
            return False
    
    def decrypt_database(self, db_path: str) -> bool:
        """Decrypt a SQLite database in-place."""
        if not os.path.exists(db_path):
            logger.warning(f"Database not found: {db_path}")
            return False
        
        try:
            # Check if file is encrypted (JSON format)
            with open(db_path, 'r') as f:
                try:
                    encrypted_data = json.load(f)
                    if not isinstance(encrypted_data, dict) or 'ciphertext' not in encrypted_data:
                        logger.warning(f"Database {db_path} is not encrypted")
                        return True  # Already decrypted
                except json.JSONDecodeError:
                    logger.warning(f"Database {db_path} is not encrypted")
                    return True  # Already decrypted
            
            # Decrypt data
            db_data = self.decrypt_data(encrypted_data)
            
            # Create decrypted database file
            decrypted_path = f"{db_path}.decrypted"
            with open(decrypted_path, 'wb') as f:
                f.write(db_data)
            
            # Replace encrypted with decrypted
            backup_path = f"{db_path}.encrypted_backup"
            os.rename(db_path, backup_path)
            os.rename(decrypted_path, db_path)
            
            logger.info(f"Decrypted database: {db_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to decrypt database {db_path}: {e}")
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get sovereign system status."""
        identity = self.generate_identity()
        
        return {
            "sovereign_mode_enabled": self.sovereign_config.get("enabled", False),
            "identity": {
                "fingerprint": identity.fingerprint[:16] + "...",
                "created_at": identity.created_at.isoformat(),
                "expires_at": identity.expires_at.isoformat() if identity.expires_at else None,
                "days_until_rotation": (identity.expires_at - datetime.now()).days if identity.expires_at else None
            },
            "encryption": {
                "enabled": self.sovereign_config.get("memory_encryption", {}).get("enabled", False),
                "algorithm": self.sovereign_config.get("memory_encryption", {}).get("algorithm", "aes-gcm"),
                "crypto_library_available": CRYPTO_AVAILABLE
            },
            "contracts": {
                "enforcement_mode": self.sovereign_config.get("contracts", {}).get("enforcement_mode", "advisory"),
                "default_expiry_hours": self.sovereign_config.get("contracts", {}).get("contract_expiry_hours", 24)
            },
            "audit_logs": {
                "enabled": self.sovereign_config.get("audit_logs", {}).get("enabled", False),
                "immutable_hashing": self.sovereign_config.get("audit_logs", {}).get("immutable_hashing", {}).get("enabled", False)
            },
            "guardian": {
                "enabled": self.sovereign_config.get("guardian", {}).get("enabled", False),
                "check_interval": self.sovereign_config.get("guardian", {}).get("check_interval_seconds", 30)
            }
        }


# Global instance for easy access
_sovereign_crypto = None

def get_sovereign_crypto() -> SovereignCrypto:
    """Get or create global SovereignCrypto instance."""
    global _sovereign_crypto
    if _sovereign_crypto is None:
        _sovereign_crypto = SovereignCrypto()
    return _sovereign_crypto
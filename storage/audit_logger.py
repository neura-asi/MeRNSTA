#!/usr/bin/env python3
"""
Immutable Audit Logger for MeRNSTA Sovereign Mode
Provides tamper-proof logging with optional IPFS/CID integration.
"""

import os
import json
import hashlib
import logging
import asyncio
import aiofiles
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass, field
import uuid

try:
    import aioipfs
    IPFS_AVAILABLE = True
except ImportError:
    IPFS_AVAILABLE = False

import yaml

logger = logging.getLogger(__name__)


@dataclass
class AuditEntry:
    """Immutable audit log entry."""
    id: str
    timestamp: datetime
    event_type: str
    agent_id: str
    data: Dict[str, Any]
    previous_hash: Optional[str] = None
    content_hash: Optional[str] = None
    ipfs_cid: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type,
            'agent_id': self.agent_id,
            'data': self.data,
            'previous_hash': self.previous_hash,
            'content_hash': self.content_hash,
            'ipfs_cid': self.ipfs_cid
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuditEntry':
        """Create instance from dictionary."""
        return cls(
            id=data['id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            event_type=data['event_type'],
            agent_id=data['agent_id'],
            data=data['data'],
            previous_hash=data.get('previous_hash'),
            content_hash=data.get('content_hash'),
            ipfs_cid=data.get('ipfs_cid')
        )
    
    def calculate_hash(self) -> str:
        """Calculate content hash for integrity verification."""
        content = {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type,
            'agent_id': self.agent_id,
            'data': self.data,
            'previous_hash': self.previous_hash
        }
        content_json = json.dumps(content, sort_keys=True)
        return hashlib.sha256(content_json.encode()).hexdigest()


class AuditLogger:
    """
    Immutable audit logging system with blockchain-like integrity.
    
    Features:
    - Cryptographic hash chaining for tamper detection
    - Optional IPFS integration for decentralized storage
    - Batch processing for efficiency
    - Rotation and compression
    - Integrity verification
    """
    
    def __init__(self, component_name: str, config_path: str = "configs/sovereign_config.yaml"):
        """Initialize audit logger for a specific component."""
        self.component_name = component_name
        self.config = self._load_config(config_path)
        self.audit_config = self.config.get("sovereign_mode", {}).get("audit_logs", {})
        
        # Setup paths
        log_path = self.audit_config.get("storage", {}).get("local_path", "./logs/sovereign_audit")
        self.log_dir = Path(log_path)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_log_file = self.log_dir / f"{component_name}.jsonl"
        self.metadata_file = self.log_dir / f"{component_name}_metadata.json"
        self.cid_file = self.log_dir / f"{component_name}_cids.jsonl"
        
        # Internal state
        self.entry_buffer: List[AuditEntry] = []
        self.last_hash: Optional[str] = None
        self.entry_count = 0
        self.ipfs_client = None
        
        # Initialize metadata
        asyncio.create_task(self._initialize_metadata())
        
        # Initialize IPFS if enabled
        if self.audit_config.get("immutable_hashing", {}).get("ipfs_integration", False):
            asyncio.create_task(self._initialize_ipfs())
        
        logger.info(f"Audit logger initialized for component: {component_name}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load audit configuration."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load audit config: {e}")
            return {}
    
    async def _initialize_metadata(self):
        """Initialize or load audit metadata."""
        if self.metadata_file.exists():
            try:
                async with aiofiles.open(self.metadata_file, 'r') as f:
                    content = await f.read()
                    metadata = json.loads(content)
                    self.last_hash = metadata.get('last_hash')
                    self.entry_count = metadata.get('entry_count', 0)
            except Exception as e:
                logger.error(f"Failed to load audit metadata: {e}")
                await self._save_metadata()
        else:
            await self._save_metadata()
    
    async def _save_metadata(self):
        """Save audit metadata."""
        metadata = {
            'component_name': self.component_name,
            'last_hash': self.last_hash,
            'entry_count': self.entry_count,
            'last_updated': datetime.now().isoformat(),
            'log_file': str(self.current_log_file),
            'config': self.audit_config
        }
        
        try:
            async with aiofiles.open(self.metadata_file, 'w') as f:
                await f.write(json.dumps(metadata, indent=2))
        except Exception as e:
            logger.error(f"Failed to save audit metadata: {e}")
    
    async def _initialize_ipfs(self):
        """Initialize IPFS client if available."""
        if not IPFS_AVAILABLE:
            logger.warning("IPFS library not available - CID generation will be local only")
            return
        
        try:
            ipfs_api_url = self.audit_config.get("immutable_hashing", {}).get("ipfs_api_url", "http://localhost:5001")
            self.ipfs_client = aioipfs.AsyncIPFS(api_url=ipfs_api_url)
            
            # Test connection
            await self.ipfs_client.version()
            logger.info("IPFS client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize IPFS client: {e}")
            self.ipfs_client = None
    
    async def log_event(self, event_data: Dict[str, Any]) -> str:
        """Log an audit event and return entry ID."""
        entry_id = str(uuid.uuid4())
        
        # Create audit entry
        entry = AuditEntry(
            id=entry_id,
            timestamp=datetime.now(),
            event_type=event_data.get('event_type', 'unknown'),
            agent_id=event_data.get('agent_id', 'system'),
            data=event_data,
            previous_hash=self.last_hash
        )
        
        # Calculate content hash
        entry.content_hash = entry.calculate_hash()
        self.last_hash = entry.content_hash
        
        # Add to buffer
        self.entry_buffer.append(entry)
        self.entry_count += 1
        
        # Check if buffer should be flushed
        batch_size = self.audit_config.get("immutable_hashing", {}).get("batch_size", 100)
        if len(self.entry_buffer) >= batch_size:
            await self._flush_buffer()
        
        logger.debug(f"Logged audit event: {entry.event_type} ({entry_id})")
        return entry_id
    
    async def _flush_buffer(self):
        """Flush buffered entries to disk and optionally IPFS."""
        if not self.entry_buffer:
            return
        
        try:
            # Write entries to log file
            async with aiofiles.open(self.current_log_file, 'a') as f:
                for entry in self.entry_buffer:
                    line = json.dumps(entry.to_dict()) + '\n'
                    await f.write(line)
            
            # Process IPFS if enabled
            if self.audit_config.get("immutable_hashing", {}).get("enabled", True):
                await self._process_ipfs_hashing()
            
            # Clear buffer
            self.entry_buffer.clear()
            
            # Save metadata
            await self._save_metadata()
            
            # Check for log rotation
            await self._check_log_rotation()
            
            logger.debug(f"Flushed {len(self.entry_buffer)} audit entries")
            
        except Exception as e:
            logger.error(f"Failed to flush audit buffer: {e}")
    
    async def _process_ipfs_hashing(self):
        """Process IPFS hashing for buffered entries."""
        if not self.entry_buffer:
            return
        
        # Create batch for hashing
        batch_data = {
            'entries': [entry.to_dict() for entry in self.entry_buffer],
            'batch_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'component': self.component_name
        }
        
        # Generate local CID
        local_cid = await self._generate_local_cid(batch_data)
        
        # Store CID mapping
        await self._store_cid_mapping(batch_data['batch_id'], local_cid, len(self.entry_buffer))
        
        # Upload to IPFS if client available
        if self.ipfs_client:
            try:
                ipfs_cid = await self._upload_to_ipfs(batch_data)
                if ipfs_cid:
                    # Update CID mapping with IPFS CID
                    await self._update_cid_mapping(batch_data['batch_id'], ipfs_cid)
                    
                    # Update entries with IPFS CID
                    for entry in self.entry_buffer:
                        entry.ipfs_cid = ipfs_cid
                        
            except Exception as e:
                logger.error(f"Failed to upload batch to IPFS: {e}")
    
    async def _generate_local_cid(self, data: Dict[str, Any]) -> str:
        """Generate local content identifier."""
        data_json = json.dumps(data, sort_keys=True)
        content_hash = hashlib.sha256(data_json.encode()).hexdigest()
        return f"local:{content_hash}"
    
    async def _upload_to_ipfs(self, data: Dict[str, Any]) -> Optional[str]:
        """Upload data to IPFS and return CID."""
        if not self.ipfs_client:
            return None
        
        try:
            data_json = json.dumps(data, sort_keys=True)
            result = await self.ipfs_client.add_bytes(data_json.encode())
            return result['Hash']
        except Exception as e:
            logger.error(f"Failed to upload to IPFS: {e}")
            return None
    
    async def _store_cid_mapping(self, batch_id: str, cid: str, entry_count: int):
        """Store CID mapping for batch."""
        mapping = {
            'batch_id': batch_id,
            'cid': cid,
            'entry_count': entry_count,
            'timestamp': datetime.now().isoformat(),
            'component': self.component_name
        }
        
        try:
            async with aiofiles.open(self.cid_file, 'a') as f:
                line = json.dumps(mapping) + '\n'
                await f.write(line)
        except Exception as e:
            logger.error(f"Failed to store CID mapping: {e}")
    
    async def _update_cid_mapping(self, batch_id: str, ipfs_cid: str):
        """Update CID mapping with IPFS CID."""
        # For simplicity, append update record
        update = {
            'batch_id': batch_id,
            'ipfs_cid': ipfs_cid,
            'update_type': 'ipfs_upload',
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            async with aiofiles.open(self.cid_file, 'a') as f:
                line = json.dumps(update) + '\n'
                await f.write(line)
        except Exception as e:
            logger.error(f"Failed to update CID mapping: {e}")
    
    async def _check_log_rotation(self):
        """Check if log file needs rotation."""
        max_size_mb = self.audit_config.get("storage", {}).get("max_size_mb", 1024)
        max_size_bytes = max_size_mb * 1024 * 1024
        
        if self.current_log_file.exists():
            current_size = self.current_log_file.stat().st_size
            
            if current_size >= max_size_bytes:
                await self._rotate_log_file()
    
    async def _rotate_log_file(self):
        """Rotate current log file."""
        rotation_count = self.audit_config.get("storage", {}).get("rotation_count", 10)
        compression = self.audit_config.get("storage", {}).get("compression", True)
        
        # Rotate existing files
        for i in range(rotation_count - 1, 0, -1):
            old_file = self.log_dir / f"{self.component_name}.{i}.jsonl"
            new_file = self.log_dir / f"{self.component_name}.{i + 1}.jsonl"
            
            if old_file.exists():
                old_file.rename(new_file)
        
        # Move current to .1
        if self.current_log_file.exists():
            rotated_file = self.log_dir / f"{self.component_name}.1.jsonl"
            self.current_log_file.rename(rotated_file)
            
            # Compress if enabled
            if compression:
                await self._compress_log_file(rotated_file)
        
        logger.info(f"Rotated audit log for component: {self.component_name}")
    
    async def _compress_log_file(self, log_file: Path):
        """Compress a rotated log file."""
        try:
            import gzip
            
            compressed_file = log_file.with_suffix('.jsonl.gz')
            
            async with aiofiles.open(log_file, 'rb') as f_in:
                content = await f_in.read()
                
            with gzip.open(compressed_file, 'wb') as f_out:
                f_out.write(content)
            
            # Remove uncompressed file
            log_file.unlink()
            
            logger.debug(f"Compressed log file: {compressed_file}")
            
        except Exception as e:
            logger.error(f"Failed to compress log file {log_file}: {e}")
    
    async def verify_integrity(self, entry_count: Optional[int] = None) -> Dict[str, Any]:
        """Verify audit log integrity."""
        if not self.current_log_file.exists():
            return {"status": "no_log_file", "verified": False}
        
        try:
            entries = []
            async with aiofiles.open(self.current_log_file, 'r') as f:
                async for line in f:
                    if line.strip():
                        entry_data = json.loads(line)
                        entries.append(AuditEntry.from_dict(entry_data))
            
            if entry_count and len(entries) != entry_count:
                return {
                    "status": "count_mismatch",
                    "verified": False,
                    "expected": entry_count,
                    "actual": len(entries)
                }
            
            # Verify hash chain
            previous_hash = None
            for i, entry in enumerate(entries):
                # Verify previous hash
                if entry.previous_hash != previous_hash:
                    return {
                        "status": "hash_chain_broken",
                        "verified": False,
                        "entry_index": i,
                        "entry_id": entry.id
                    }
                
                # Verify content hash
                calculated_hash = entry.calculate_hash()
                if entry.content_hash != calculated_hash:
                    return {
                        "status": "content_hash_mismatch",
                        "verified": False,
                        "entry_index": i,
                        "entry_id": entry.id
                    }
                
                previous_hash = entry.content_hash
            
            return {
                "status": "verified",
                "verified": True,
                "total_entries": len(entries),
                "last_hash": previous_hash
            }
            
        except Exception as e:
            logger.error(f"Integrity verification failed: {e}")
            return {"status": "verification_error", "verified": False, "error": str(e)}
    
    async def get_entries(self, event_type: Optional[str] = None,
                         agent_id: Optional[str] = None,
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None,
                         limit: int = 100) -> List[AuditEntry]:
        """Retrieve audit entries with filtering."""
        if not self.current_log_file.exists():
            return []
        
        try:
            entries = []
            count = 0
            
            async with aiofiles.open(self.current_log_file, 'r') as f:
                async for line in f:
                    if count >= limit:
                        break
                    
                    if line.strip():
                        entry_data = json.loads(line)
                        entry = AuditEntry.from_dict(entry_data)
                        
                        # Apply filters
                        if event_type and entry.event_type != event_type:
                            continue
                        if agent_id and entry.agent_id != agent_id:
                            continue
                        if start_time and entry.timestamp < start_time:
                            continue
                        if end_time and entry.timestamp > end_time:
                            continue
                        
                        entries.append(entry)
                        count += 1
            
            return entries
            
        except Exception as e:
            logger.error(f"Failed to retrieve entries: {e}")
            return []
    
    async def close(self):
        """Close audit logger and flush any remaining entries."""
        await self._flush_buffer()
        
        if self.ipfs_client:
            try:
                await self.ipfs_client.close()
            except Exception as e:
                logger.error(f"Failed to close IPFS client: {e}")
        
        logger.info(f"Audit logger closed for component: {self.component_name}")
    
    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, 'entry_buffer') and self.entry_buffer:
            # Cannot use async in __del__, so just log warning
            logger.warning(f"Audit logger deleted with {len(self.entry_buffer)} unflushed entries")


# Global audit logger registry
_audit_loggers = {}

def get_audit_logger(component_name: str) -> AuditLogger:
    """Get or create audit logger for component."""
    if component_name not in _audit_loggers:
        _audit_loggers[component_name] = AuditLogger(component_name)
    return _audit_loggers[component_name]
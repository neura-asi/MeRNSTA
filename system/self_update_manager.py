#!/usr/bin/env python3
"""
Autonomous Self-Update Manager for MeRNSTA Sovereign Mode
Provides cryptographically signed autonomous updates with rollback capabilities.
"""

import os
import sys
import json
import hashlib
import logging
import asyncio
import aiohttp
import tempfile
import shutil
import subprocess
import zipfile
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
import semver
try:
    import git  # type: ignore
except Exception:  # GitPython optional
    git = None

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from system.sovereign_crypto import get_sovereign_crypto
from storage.audit_logger import get_audit_logger

logger = logging.getLogger(__name__)


@dataclass
class UpdatePackage:
    """Represents a signed update package."""
    version: str
    description: str
    package_url: str
    package_hash: str
    signature: str
    signer_key: str
    release_date: datetime
    compatibility: Dict[str, Any]
    rollback_instructions: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'version': self.version,
            'description': self.description,
            'package_url': self.package_url,
            'package_hash': self.package_hash,
            'signature': self.signature,
            'signer_key': self.signer_key,
            'release_date': self.release_date.isoformat(),
            'compatibility': self.compatibility,
            'rollback_instructions': self.rollback_instructions
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UpdatePackage':
        return cls(
            version=data['version'],
            description=data['description'],
            package_url=data['package_url'],
            package_hash=data['package_hash'],
            signature=data['signature'],
            signer_key=data['signer_key'],
            release_date=datetime.fromisoformat(data['release_date']),
            compatibility=data['compatibility'],
            rollback_instructions=data.get('rollback_instructions')
        )


@dataclass
class SignedGoal:
    """Represents a cryptographically signed autonomous goal."""
    goal_id: str
    description: str
    target_version: Optional[str]
    conditions: Dict[str, Any]  # Conditions that must be met
    actions: List[Dict[str, Any]]  # Actions to take
    priority: int  # Higher = more important
    created_at: datetime
    expires_at: Optional[datetime]
    signature: str
    signer_key: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'goal_id': self.goal_id,
            'description': self.description,
            'target_version': self.target_version,
            'conditions': self.conditions,
            'actions': self.actions,
            'priority': self.priority,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'signature': self.signature,
            'signer_key': self.signer_key
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SignedGoal':
        return cls(
            goal_id=data['goal_id'],
            description=data['description'],
            target_version=data.get('target_version'),
            conditions=data['conditions'],
            actions=data['actions'],
            priority=data['priority'],
            created_at=datetime.fromisoformat(data['created_at']),
            expires_at=datetime.fromisoformat(data['expires_at']) if data.get('expires_at') else None,
            signature=data['signature'],
            signer_key=data['signer_key']
        )
    
    def is_expired(self) -> bool:
        """Check if goal has expired."""
        return self.expires_at is not None and datetime.now() > self.expires_at
    
    def get_signing_payload(self) -> str:
        """Get payload for signature verification."""
        payload = {
            'goal_id': self.goal_id,
            'description': self.description,
            'target_version': self.target_version,
            'conditions': self.conditions,
            'actions': self.actions,
            'priority': self.priority,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None
        }
        return json.dumps(payload, sort_keys=True)


class SelfUpdateManager:
    """
    Manages autonomous self-updates with cryptographic verification.
    
    Features:
    - Signed update packages with verification
    - Autonomous goal execution based on signed instructions
    - Rollback capability for failed updates
    - Version compatibility checking
    - Update source validation
    """
    
    def __init__(self):
        """Initialize self-update manager."""
        self.crypto = get_sovereign_crypto()
        self.config = self.crypto.sovereign_config.get("self_update", {})
        self.audit_logger = get_audit_logger("self_update")
        
        # Current state
        self.current_version = self._get_current_version()
        self.update_history: List[Dict[str, Any]] = []
        self.signed_goals: List[SignedGoal] = []
        self.trusted_signers = set(self.config.get("trusted_signers", []))
        
        # Paths
        self.updates_dir = Path("./updates")
        self.updates_dir.mkdir(exist_ok=True)
        self.backups_dir = Path("./backups/versions")
        self.backups_dir.mkdir(parents=True, exist_ok=True)
        
        # State
        self._monitoring = False
        self._last_check = None
        
        logger.info(f"Self-update manager initialized - current version: {self.current_version}")
    
    def _get_current_version(self) -> str:
        """Get current MeRNSTA version."""
        try:
            # Try to read from version file
            version_file = Path("VERSION")
            if version_file.exists():
                return version_file.read_text().strip()
            
            # Try to get from git (optional)
            if git is not None:
                try:
                    repo = git.Repo(".")
                    tags = sorted(repo.tags, key=lambda t: t.commit.committed_datetime, reverse=True)
                    if tags:
                        return str(tags[0])
                except Exception:
                    pass
            
            # Default version
            return "0.7.0"
            
        except Exception as e:
            logger.error(f"Failed to get current version: {e}")
            return "unknown"
    
    async def start_monitoring(self):
        """Start autonomous update monitoring."""
        if self._monitoring:
            logger.warning("Update monitoring already running")
            return
        
        if not self.config.get("enabled", False):
            logger.info("Self-update disabled in configuration")
            return
        
        self._monitoring = True
        
        # Start monitoring loop
        asyncio.create_task(self._monitoring_loop())
        
        # Load existing goals
        await self._load_signed_goals()
        
        await self.audit_logger.log_event({
            "event_type": "update_monitoring_started",
            "agent_id": "self_update",
            "current_version": self.current_version,
            "check_interval_hours": self.config.get("update_check_interval_hours", 6)
        })
        
        logger.info("Self-update monitoring started")
    
    async def stop_monitoring(self):
        """Stop autonomous update monitoring."""
        self._monitoring = False
        
        await self.audit_logger.log_event({
            "event_type": "update_monitoring_stopped",
            "agent_id": "self_update"
        })
        
        logger.info("Self-update monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop for autonomous updates."""
        check_interval = self.config.get("update_check_interval_hours", 6) * 3600
        
        while self._monitoring:
            try:
                await self._perform_update_cycle()
                self._last_check = datetime.now()
            except Exception as e:
                logger.error(f"Update monitoring cycle failed: {e}")
            
            await asyncio.sleep(check_interval)
    
    async def _perform_update_cycle(self):
        """Perform one complete update cycle."""
        # Check for available updates
        available_updates = await self._check_for_updates()
        
        # Process signed goals
        await self._process_signed_goals()
        
        # Evaluate updates based on configuration and goals
        for update in available_updates:
            if await self._should_apply_update(update):
                await self._apply_update(update)
    
    async def _check_for_updates(self) -> List[UpdatePackage]:
        """Check all configured sources for available updates."""
        updates = []
        
        # Check primary source
        primary_source = self.config.get("update_sources", {}).get("primary")
        if primary_source:
            updates.extend(await self._check_update_source(primary_source))
        
        # Check fallback source
        fallback_source = self.config.get("update_sources", {}).get("fallback")
        if fallback_source and not updates:  # Only if primary failed
            updates.extend(await self._check_update_source(fallback_source))
        
        # Check local git repository
        if self.config.get("update_sources", {}).get("local_repo", False):
            updates.extend(await self._check_local_repo())
        
        # Filter and sort updates
        valid_updates = []
        for update in updates:
            if await self._verify_update_package(update):
                valid_updates.append(update)
        
        # Sort by version (newest first)
        valid_updates.sort(key=lambda u: semver.VersionInfo.parse(u.version), reverse=True)
        
        return valid_updates
    
    async def _check_update_source(self, source_url: str) -> List[UpdatePackage]:
        """Check a specific update source."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{source_url}/updates.json") as response:
                    if response.status == 200:
                        data = await response.json()
                        return [UpdatePackage.from_dict(update) for update in data.get("updates", [])]
        except Exception as e:
            logger.error(f"Failed to check update source {source_url}: {e}")
        
        return []
    
    async def _check_local_repo(self) -> List[UpdatePackage]:
        """Check local git repository for updates."""
        if git is None:
            return []
        try:
            repo = git.Repo(".")
            
            # Fetch latest changes
            try:
                repo.remotes.origin.fetch()
            except Exception:
                pass
            
            # Get tags newer than current version
            current_ver = semver.VersionInfo.parse(self.current_version)
            updates = []
            
            for tag in repo.tags:
                try:
                    tag_ver = semver.VersionInfo.parse(str(tag))
                    if tag_ver > current_ver:
                        # Create update package for this tag
                        update = UpdatePackage(
                            version=str(tag),
                            description=f"Git tag update to {tag}",
                            package_url="local",
                            package_hash=str(getattr(tag, 'commit', 'unknown')),
                            signature="local_git",
                            signer_key="local",
                            release_date=datetime.fromtimestamp(getattr(getattr(tag, 'commit', None), 'committed_date', datetime.now().timestamp())),
                            compatibility={"git": True}
                        )
                        updates.append(update)
                except Exception:
                    continue
            
            return updates
            
        except Exception as e:
            logger.error(f"Failed to check local repository: {e}")
            return []
    
    async def _verify_update_package(self, update: UpdatePackage) -> bool:
        """Verify an update package signature and integrity."""
        try:
            # Check if signer is trusted
            if update.signer_key not in self.trusted_signers and update.signer_key != "local":
                logger.warning(f"Update from untrusted signer: {update.signer_key}")
                return False
            
            # For local git updates, skip signature verification
            if update.package_url == "local":
                return True
            
            # Verify signature (TODO: implement proper signature verification)
            # For now, just check if signature exists and is non-empty
            if not update.signature or len(update.signature) < 10:
                logger.error(f"Invalid signature for update {update.version}")
                return False
            
            # Check version format
            try:
                semver.VersionInfo.parse(update.version)
            except ValueError:
                logger.error(f"Invalid version format: {update.version}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Update verification failed for {update.version}: {e}")
            return False
    
    async def _should_apply_update(self, update: UpdatePackage) -> bool:
        """Determine if an update should be applied based on configuration and goals."""
        current_ver = semver.VersionInfo.parse(self.current_version)
        update_ver = semver.VersionInfo.parse(update.version)
        
        # Don't downgrade
        if update_ver <= current_ver:
            return False
        
        # Check auto-apply settings
        version_diff = update_ver.compare(current_ver)
        
        if version_diff == 1:  # Major version
            # Never auto-apply major versions
            return False
        elif version_diff == 2:  # Minor version
            if not self.config.get("auto_apply_minor", False):
                return False
        elif version_diff == 3:  # Patch version
            if not self.config.get("auto_apply_patch", True):
                return False
        
        # Check signed goals
        for goal in self.signed_goals:
            if goal.is_expired():
                continue
            
            if goal.target_version and goal.target_version == update.version:
                if await self._evaluate_goal_conditions(goal):
                    logger.info(f"Update {update.version} approved by signed goal {goal.goal_id}")
                    return True
        
        # Check compatibility
        if not await self._check_compatibility(update):
            return False
        
        return True
    
    async def _evaluate_goal_conditions(self, goal: SignedGoal) -> bool:
        """Evaluate if goal conditions are met."""
        try:
            conditions = goal.conditions
            
            # Check time-based conditions
            if "after_time" in conditions:
                after_time = datetime.fromisoformat(conditions["after_time"])
                if datetime.now() < after_time:
                    return False
            
            # Check system state conditions
            if "min_uptime_hours" in conditions:
                # TODO: Check actual system uptime
                pass
            
            if "require_encrypted_memory" in conditions:
                # Check if memory encryption is active
                from storage.memory_encryption import get_memory_encryption_manager
                manager = get_memory_encryption_manager()
                status = manager.get_encryption_status()
                if not status.get("encryption_enabled"):
                    return False
            
            # All conditions met
            return True
            
        except Exception as e:
            logger.error(f"Failed to evaluate goal conditions: {e}")
            return False
    
    async def _check_compatibility(self, update: UpdatePackage) -> bool:
        """Check if update is compatible with current system."""
        compatibility = update.compatibility
        
        # Check Python version
        if "python_version" in compatibility:
            required_python = compatibility["python_version"]
            current_python = f"{sys.version_info.major}.{sys.version_info.minor}"
            if current_python < required_python:
                logger.warning(f"Update {update.version} requires Python {required_python}, have {current_python}")
                return False
        
        # Check OS compatibility
        if "os" in compatibility:
            import platform
            required_os = compatibility["os"]
            current_os = platform.system().lower()
            if current_os not in [os.lower() for os in required_os]:
                logger.warning(f"Update {update.version} not compatible with {current_os}")
                return False
        
        return True
    
    async def _apply_update(self, update: UpdatePackage) -> bool:
        """Apply an update package."""
        logger.info(f"Applying update: {update.version}")
        
        try:
            # Create backup
            backup_path = await self._create_backup()
            
            # Download and verify package
            package_path = await self._download_package(update)
            if not package_path:
                return False
            
            # Verify package hash
            if not await self._verify_package_hash(package_path, update.package_hash):
                logger.error(f"Package hash verification failed for {update.version}")
                return False
            
            # Apply the update
            success = await self._install_update(package_path, update)
            
            if success:
                # Update version tracking
                self.current_version = update.version
                self.update_history.append({
                    "version": update.version,
                    "applied_at": datetime.now().isoformat(),
                    "backup_path": str(backup_path),
                    "package_hash": update.package_hash
                })
                
                # Log successful update
                await self.audit_logger.log_event({
                    "event_type": "update_applied",
                    "agent_id": "self_update",
                    "update": update.to_dict(),
                    "backup_path": str(backup_path)
                })
                
                logger.info(f"Successfully applied update {update.version}")
                
                # Execute post-update actions if specified in goals
                await self._execute_post_update_actions(update)
                
            else:
                # Rollback on failure
                logger.error(f"Update {update.version} failed, initiating rollback")
                await self._rollback_update(backup_path)
            
            # Cleanup
            if package_path and package_path.exists():
                package_path.unlink()
            
            return success
            
        except Exception as e:
            logger.error(f"Update application failed: {e}")
            return False
    
    async def _create_backup(self) -> Path:
        """Create a backup of the current system."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backups_dir / f"backup_{self.current_version}_{timestamp}"
        backup_path.mkdir(exist_ok=True)
        
        # Copy critical files
        critical_files = [
            "configs/config.yaml",
            "configs/sovereign_config.yaml",
            "requirements.txt",
            "agents/",
            "system/",
            "storage/",
            "cli/"
        ]
        
        for file_path in critical_files:
            src = Path(file_path)
            if src.exists():
                dst = backup_path / file_path
                if src.is_dir():
                    shutil.copytree(src, dst, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
                else:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
        
        logger.info(f"Created backup at {backup_path}")
        return backup_path
    
    async def _download_package(self, update: UpdatePackage) -> Optional[Path]:
        """Download update package."""
        if update.package_url == "local":
            return None  # Local git updates don't need downloading
        
        try:
            package_path = self.updates_dir / f"update_{update.version}.zip"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(update.package_url) as response:
                    if response.status == 200:
                        with open(package_path, 'wb') as f:
                            async for chunk in response.content.iter_chunked(8192):
                                f.write(chunk)
                        
                        logger.info(f"Downloaded update package: {package_path}")
                        return package_path
                    else:
                        logger.error(f"Failed to download package: HTTP {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"Package download failed: {e}")
            return None
    
    async def _verify_package_hash(self, package_path: Path, expected_hash: str) -> bool:
        """Verify package integrity using hash."""
        try:
            hash_algo = self.config.get("validation", {}).get("hash_algorithm", "sha256")
            
            hasher = hashlib.new(hash_algo)
            with open(package_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
            
            calculated_hash = hasher.hexdigest()
            return calculated_hash == expected_hash
            
        except Exception as e:
            logger.error(f"Hash verification failed: {e}")
            return False
    
    async def _install_update(self, package_path: Optional[Path], update: UpdatePackage) -> bool:
        """Install the update package."""
        try:
            if update.package_url == "local":
                # Git-based update
                return await self._install_git_update(update)
            else:
                # Package-based update
                return await self._install_package_update(package_path, update)
                
        except Exception as e:
            logger.error(f"Update installation failed: {e}")
            return False
    
    async def _install_git_update(self, update: UpdatePackage) -> bool:
        """Install update from git repository."""
        try:
            repo = git.Repo(".")
            
            # Checkout the target version
            repo.git.checkout(update.version)
            
            # Update dependencies if requirements changed
            requirements_file = Path("requirements.txt")
            if requirements_file.exists():
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
                ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    logger.error(f"Failed to update dependencies: {result.stderr}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Git update failed: {e}")
            return False
    
    async def _install_package_update(self, package_path: Path, update: UpdatePackage) -> bool:
        """Install update from package file."""
        try:
            # Extract package
            extract_dir = self.updates_dir / f"extract_{update.version}"
            extract_dir.mkdir(exist_ok=True)
            
            with zipfile.ZipFile(package_path, 'r') as zip_file:
                zip_file.extractall(extract_dir)
            
            # Apply update files
            update_files = list(extract_dir.rglob("*"))
            for file_path in update_files:
                if file_path.is_file():
                    relative_path = file_path.relative_to(extract_dir)
                    target_path = Path(relative_path)
                    
                    # Ensure target directory exists
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Copy file
                    shutil.copy2(file_path, target_path)
            
            # Run update script if present
            update_script = extract_dir / "update.py"
            if update_script.exists():
                result = subprocess.run([
                    sys.executable, str(update_script)
                ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    logger.error(f"Update script failed: {result.stderr}")
                    return False
            
            # Cleanup
            shutil.rmtree(extract_dir)
            
            return True
            
        except Exception as e:
            logger.error(f"Package update failed: {e}")
            return False
    
    async def _execute_post_update_actions(self, update: UpdatePackage):
        """Execute post-update actions specified in signed goals."""
        for goal in self.signed_goals:
            if goal.target_version == update.version:
                for action in goal.actions:
                    await self._execute_goal_action(action, goal)
    
    async def _execute_goal_action(self, action: Dict[str, Any], goal: SignedGoal):
        """Execute a specific goal action."""
        try:
            action_type = action.get("type")
            
            if action_type == "restart_service":
                # TODO: Implement service restart
                logger.info(f"Goal {goal.goal_id}: Restart service requested")
                
            elif action_type == "encrypt_memory":
                # Encrypt memory if not already encrypted
                from storage.memory_encryption import get_memory_encryption_manager
                manager = get_memory_encryption_manager()
                manager.encrypt_all_databases()
                logger.info(f"Goal {goal.goal_id}: Memory encryption activated")
                
            elif action_type == "generate_identity":
                # Generate new identity
                self.crypto.generate_identity(force_new=True)
                logger.info(f"Goal {goal.goal_id}: New identity generated")
                
            elif action_type == "notify":
                # Send notification
                message = action.get("message", f"Goal {goal.goal_id} completed")
                logger.info(f"Goal {goal.goal_id}: {message}")
            
            # Log action execution
            await self.audit_logger.log_event({
                "event_type": "goal_action_executed",
                "agent_id": "self_update",
                "goal_id": goal.goal_id,
                "action": action
            })
            
        except Exception as e:
            logger.error(f"Failed to execute goal action: {e}")
    
    async def _rollback_update(self, backup_path: Path) -> bool:
        """Rollback to previous version using backup."""
        try:
            logger.warning(f"Rolling back to backup: {backup_path}")
            
            # Restore files from backup
            for item in backup_path.rglob("*"):
                if item.is_file():
                    relative_path = item.relative_to(backup_path)
                    target_path = Path(relative_path)
                    
                    # Ensure target directory exists
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Restore file
                    shutil.copy2(item, target_path)
            
            # Log rollback
            await self.audit_logger.log_event({
                "event_type": "update_rollback",
                "agent_id": "self_update",
                "backup_path": str(backup_path),
                "rollback_successful": True
            })
            
            logger.info(f"Successfully rolled back to backup")
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    async def add_signed_goal(self, goal: SignedGoal) -> bool:
        """Add a signed goal for autonomous execution."""
        try:
            # Verify goal signature
            if not await self._verify_goal_signature(goal):
                logger.error(f"Invalid signature for goal {goal.goal_id}")
                return False
            
            # Check if signer is trusted
            if goal.signer_key not in self.trusted_signers:
                logger.warning(f"Goal from untrusted signer: {goal.signer_key}")
                return False
            
            # Add goal
            self.signed_goals.append(goal)
            
            # Save goals
            await self._save_signed_goals()
            
            # Log goal addition
            await self.audit_logger.log_event({
                "event_type": "signed_goal_added",
                "agent_id": "self_update",
                "goal": goal.to_dict()
            })
            
            logger.info(f"Added signed goal: {goal.goal_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add signed goal: {e}")
            return False
    
    async def _verify_goal_signature(self, goal: SignedGoal) -> bool:
        """Verify a signed goal's signature."""
        try:
            # TODO: Implement proper signature verification
            # For now, just check if signature exists
            return len(goal.signature) > 10
            
        except Exception as e:
            logger.error(f"Goal signature verification failed: {e}")
            return False
    
    async def _process_signed_goals(self):
        """Process all signed goals and execute applicable ones."""
        for goal in self.signed_goals[:]:  # Copy list to allow modification
            if goal.is_expired():
                self.signed_goals.remove(goal)
                continue
            
            if await self._evaluate_goal_conditions(goal):
                await self._execute_goal_actions(goal)
    
    async def _execute_goal_actions(self, goal: SignedGoal):
        """Execute all actions for a goal."""
        for action in goal.actions:
            await self._execute_goal_action(action, goal)
    
    async def _save_signed_goals(self):
        """Save signed goals to file."""
        goals_file = self.updates_dir / "signed_goals.json"
        goals_data = [goal.to_dict() for goal in self.signed_goals]
        
        with open(goals_file, 'w') as f:
            json.dump(goals_data, f, indent=2)
    
    async def _load_signed_goals(self):
        """Load signed goals from file."""
        goals_file = self.updates_dir / "signed_goals.json"
        
        if goals_file.exists():
            try:
                with open(goals_file, 'r') as f:
                    goals_data = json.load(f)
                
                for goal_data in goals_data:
                    goal = SignedGoal.from_dict(goal_data)
                    if not goal.is_expired():
                        self.signed_goals.append(goal)
                
                logger.info(f"Loaded {len(self.signed_goals)} signed goals")
                
            except Exception as e:
                logger.error(f"Failed to load signed goals: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get self-update system status."""
        return {
            "enabled": self.config.get("enabled", False),
            "monitoring": self._monitoring,
            "current_version": self.current_version,
            "last_check": self._last_check.isoformat() if self._last_check else None,
            "check_interval_hours": self.config.get("update_check_interval_hours", 6),
            "auto_apply_minor": self.config.get("auto_apply_minor", False),
            "auto_apply_patch": self.config.get("auto_apply_patch", True),
            "trusted_signers": list(self.trusted_signers),
            "signed_goals": len(self.signed_goals),
            "update_history": len(self.update_history),
            "recent_updates": self.update_history[-5:] if self.update_history else []
        }


# Global self-update manager
_self_update_manager = None

def get_self_update_manager() -> SelfUpdateManager:
    """Get or create global self-update manager."""
    global _self_update_manager
    if _self_update_manager is None:
        _self_update_manager = SelfUpdateManager()
    return _self_update_manager
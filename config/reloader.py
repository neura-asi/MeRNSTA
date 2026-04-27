#!/usr/bin/env python3
"""
Hot-reload configuration system for MeRNSTA.
Watches for configuration changes and notifies subscribers.
"""

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

import yaml
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger(__name__)


class ConfigReloader(FileSystemEventHandler):
    """Watches for configuration file changes and reloads settings."""

    def __init__(self, config_paths: List[str]):
        self.config_paths = [Path(p) for p in config_paths]
        self.callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self.last_modified = {path: 0 for path in self.config_paths}
        self.lock = threading.Lock()

        # Ensure config files exist
        for config_path in self.config_paths:
            if not config_path.exists():
                logger.warning(f"Config file {config_path} does not exist")

    def on_modified(self, event):
        """Handle file modification events."""
        if event.is_directory:
            return

        file_path = Path(event.src_path)
        if file_path in self.config_paths:
            # Debounce rapid changes
            current_time = time.time()
            if current_time - self.last_modified.get(file_path, 0) < 1.0:
                return

            self.last_modified[file_path] = current_time
            logger.info(f"Config file {file_path} modified, reloading...")
            self.reload_config()

    def reload_config(self):
        """Reload configuration and notify subscribers."""
        try:
            with self.lock:
                # Reload environment settings
                from .environment import reload_settings

                new_settings = reload_settings()

                # Load YAML config if it exists
                yaml_config = {}
                yaml_path = Path("configs/config.yaml")
                if yaml_path.exists():
                    with open(yaml_path, "r") as f:
                        yaml_config = yaml.safe_load(f) or {}

                # Combine settings
                combined_config = {"settings": new_settings.dict(), "yaml": yaml_config}

                # Notify all subscribers
                for callback in self.callbacks:
                    try:
                        callback(combined_config)
                    except Exception as e:
                        logger.error(f"Error in config reload callback: {e}")

                logger.info("Configuration reloaded successfully")

        except Exception as e:
            logger.error(f"Error reloading configuration: {e}")

    def subscribe(self, callback: Callable[[Dict[str, Any]], None]):
        """Subscribe to configuration change notifications."""
        with self.lock:
            self.callbacks.append(callback)
        logger.info(f"Added config reload subscriber: {callback.__name__}")

    def unsubscribe(self, callback: Callable[[Dict[str, Any]], None]):
        """Unsubscribe from configuration change notifications."""
        with self.lock:
            if callback in self.callbacks:
                self.callbacks.remove(callback)
                logger.info(f"Removed config reload subscriber: {callback.__name__}")


class ConfigManager:
    """Manages configuration reloading and provides access to current settings."""

    def __init__(self, config_paths: List[str] = None):
        if config_paths is None:
            config_paths = [".env", "configs/config.yaml"]

        self.reloader = ConfigReloader(config_paths)
        self.observer = Observer()
        self.observer.schedule(self.reloader, path=".", recursive=False)
        self._started = False

    def start(self):
        """Start watching for configuration changes."""
        if not self._started:
            self.observer.start()
            self._started = True
            logger.info("Configuration watcher started")

    def stop(self):
        """Stop watching for configuration changes."""
        if self._started:
            self.observer.stop()
            self.observer.join()
            self._started = False
            logger.info("Configuration watcher stopped")

    def subscribe(self, callback: Callable[[Dict[str, Any]], None]):
        """Subscribe to configuration changes."""
        self.reloader.subscribe(callback)

    def unsubscribe(self, callback: Callable[[Dict[str, Any]], None]):
        """Unsubscribe from configuration changes."""
        self.reloader.unsubscribe(callback)

    def get_current_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        from .environment import get_settings

        settings = get_settings()

        yaml_config = {}
        yaml_path = Path("configs/config.yaml")
        if yaml_path.exists():
            with open(yaml_path, "r") as f:
                yaml_config = yaml.safe_load(f) or {}

        return {"settings": settings.dict(), "yaml": yaml_config}


# Global config manager instance
config_manager = ConfigManager()


def get_config_manager() -> ConfigManager:
    """Get the global config manager instance."""
    return config_manager


def start_config_watcher():
    """Start the configuration watcher."""
    config_manager.start()


def stop_config_watcher():
    """Stop the configuration watcher."""
    config_manager.stop()


# Example usage:
# def on_config_change(config: Dict[str, Any]):
#     print(f"Config changed: {config}")
#
# config_manager.subscribe(on_config_change)
# start_config_watcher()

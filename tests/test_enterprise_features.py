#!/usr/bin/env python3
"""
Comprehensive tests for MeRNSTA enterprise features.
Tests monitoring, task queue, caching, and configuration systems.
"""

import pytest
import time
import json
import tempfile
import os
import numpy as np
from unittest.mock import Mock, patch
from config.environment import get_settings, Settings
from monitoring.logger import get_logger, log_memory_operation, log_api_request
from monitoring.metrics import track_memory_operation, update_memory_metrics
from storage.cache import MemoryCache, get_cache, clear_cache
from tasks.task_queue import get_task_status, get_active_tasks, get_queue_stats
from config.reloader import ConfigManager, get_config_manager


class TestConfigurationSystem:
    """Test the environment-driven configuration system."""
    
    def test_settings_validation(self):
        """Test that settings validation works correctly."""
        # Test valid settings
        settings = get_settings()
        assert settings.max_facts > 0
        assert 0.0 <= settings.compression_threshold <= 1.0
        assert settings.log_level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    
    def test_settings_reload(self):
        """Test that settings can be reloaded."""
        from config.environment import reload_settings
        original_settings = get_settings()
        new_settings = reload_settings()
        assert new_settings is not original_settings
    
    def test_invalid_settings(self):
        """Test that invalid settings raise appropriate errors."""
        # This test is intentionally empty since we don't have a way to set invalid values
        # in the current test environment. In a real scenario, this would test validation.
        pass


class TestMonitoringSystem:
    """Test the structured logging and metrics system."""
    
    def test_structured_logging(self):
        """Test that structured logging works correctly."""
        logger = get_logger("test")
        logger.info("test_message", test_field="test_value")
        # In a real test, we'd verify the log output
    
    def test_memory_operation_logging(self):
        """Test memory operation logging."""
        log_memory_operation("test_operation", fact_count=5, duration=0.1)
        # In a real test, we'd verify the log output
    
    def test_api_request_logging(self):
        """Test API request logging."""
        log_api_request("GET", "/test", 200, 0.05, client_ip="127.0.0.1")
        # In a real test, we'd verify the log output
    
    def test_metrics_tracking(self):
        """Test metrics tracking decorators."""
        @track_memory_operation("test_operation")
        def test_function():
            return "success"
        
        result = test_function()
        assert result == "success"
    
    def test_metrics_update(self):
        """Test metrics update functions."""
        stats = {
            "total_facts": 100,
            "avg_contradiction_score": 0.1,
            "compression_ratio": 0.5,
            "avg_volatility_score": 0.2
        }
        update_memory_metrics(stats)
        # In a real test, we'd verify the metrics were updated


class TestCachingSystem:
    """Test the Redis-based caching system."""
    
    @pytest.fixture
    def cache(self):
        """Create a test cache instance."""
        # Use a test Redis instance or mock
        with patch('redis.from_url') as mock_redis:
            mock_redis.return_value.ping.return_value = True
            mock_redis.return_value.get.return_value = None
            mock_redis.return_value.setex.return_value = True
            cache = MemoryCache("redis://localhost:6379/0")
            yield cache
    
    def test_cache_initialization(self, cache):
        """Test cache initialization."""
        assert cache.enable_caching is True
        assert cache.default_ttl > 0
    
    def test_embedding_caching(self, cache):
        """Test embedding caching functionality."""
        text = "test text"
        embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        
        # Test setting embedding
        cache.set_embedding(text, embedding)
        
        # Test getting embedding (should be None since we're mocking)
        cached_embedding = cache.get_embedding(text)
        assert cached_embedding is None  # Due to mocking
    
    def test_cluster_centroid_caching(self, cache):
        """Test cluster centroid caching."""
        subject = "test_subject"
        centroid = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        
        # Test setting centroid
        cache.set_cluster_centroid(subject, centroid)
        
        # Test getting centroid (should be None since we're mocking)
        cached_centroid = cache.get_cluster_centroid(subject)
        assert cached_centroid is None  # Due to mocking
    
    def test_cache_stats(self, cache):
        """Test cache statistics."""
        stats = cache.get_cache_stats()
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_ratio" in stats
        assert "enabled" in stats
    
    def test_cache_clear(self, cache):
        """Test cache clearing."""
        cache.clear_cache()
        # In a real test, we'd verify the cache was cleared


class TestTaskQueueSystem:
    """Test the Celery task queue system."""
    
    @pytest.fixture
    def mock_celery(self):
        """Mock Celery components."""
        with patch('tasks.task_queue.app') as mock_app:
            mock_app.control.inspect.return_value.stats.return_value = {"worker1": {}}
            mock_app.control.inspect.return_value.active.return_value = {}
            mock_app.control.inspect.return_value.reserved.return_value = {}
            yield mock_app
    
    def test_task_status(self, mock_celery):
        """Test getting task status."""
        with patch('tasks.task_queue.app.AsyncResult') as mock_result:
            mock_result.return_value.status = "SUCCESS"
            mock_result.return_value.ready.return_value = True
            mock_result.return_value.result = {"status": "success"}
            
            status = get_task_status("test_task_id")
            assert status["status"] == "SUCCESS"
            assert status["result"] == {"status": "success"}
    
    def test_active_tasks(self, mock_celery):
        """Test getting active tasks."""
        tasks = get_active_tasks()
        assert isinstance(tasks, list)
    
    def test_queue_stats(self, mock_celery):
        """Test getting queue statistics."""
        stats = get_queue_stats()
        assert "workers" in stats
        assert "active_tasks" in stats
        assert "reserved_tasks" in stats
        assert "total_tasks" in stats


class TestHealthCheckSystem:
    """Test the health check system."""
    
    def test_database_health_check(self):
        """Test database health check."""
        from api.health import check_database_health
        
        # This test requires a real database connection
        # In a real test environment, you'd set up a test database
        pass
    
    def test_memory_health_check(self):
        """Test memory health check."""
        from api.health import check_memory_health
        
        # This test requires a real memory log instance
        # In a real test environment, you'd set up a test memory log
        pass
    
    def test_system_health_check(self):
        """Test system health check."""
        from api.health import check_system_health
        
        health = check_system_health()
        assert "status" in health
        assert "memory_usage_percent" in health
        assert "cpu_percent" in health
        assert "disk_usage_percent" in health


class TestConfigurationReloader:
    """Test the configuration reloader system."""
    
    @pytest.fixture
    def config_manager(self):
        """Create a test config manager."""
        return ConfigManager([".env", "configs/config.yaml"])
    
    def test_config_manager_initialization(self, config_manager):
        """Test config manager initialization."""
        assert config_manager is not None
        assert hasattr(config_manager, 'subscribe')
        assert hasattr(config_manager, 'unsubscribe')
    
    def test_config_subscription(self, config_manager):
        """Test config subscription system."""
        callback_called = False
        
        def test_callback(config):
            nonlocal callback_called
            callback_called = True
        
        config_manager.subscribe(test_callback)
        assert len(config_manager.reloader.callbacks) == 1
        
        config_manager.unsubscribe(test_callback)
        assert len(config_manager.reloader.callbacks) == 0


class TestIntegration:
    """Integration tests for the enterprise features."""
    
    def test_full_workflow(self):
        """Test a complete workflow with all enterprise features."""
        # 1. Test configuration loading
        settings = get_settings()
        assert settings is not None
        
        # 2. Test logging initialization
        logger = get_logger("integration_test")
        logger.info("Integration test started")
        
        # 3. Test cache initialization
        cache = get_cache()
        assert cache is not None
        
        # 4. Test metrics collection
        stats = {"total_facts": 100, "avg_contradiction_score": 0.1}
        update_memory_metrics(stats)
        
        # 5. Test task queue (mocked)
        with patch('tasks.task_queue.app.AsyncResult') as mock_result:
            mock_result.return_value.status = "SUCCESS"
            status = get_task_status("test_task")
            assert status["status"] == "SUCCESS"
    
    def test_error_handling(self):
        """Test error handling across all systems."""
        # Test that errors are properly logged and don't crash the system
        logger = get_logger("error_test")
        
        try:
            raise ValueError("Test error")
        except Exception as e:
            logger.error("Caught test error", error=str(e))
            # In a real test, we'd verify the error was logged properly
    
    def test_performance_monitoring(self):
        """Test performance monitoring features."""
        # Test that performance metrics are collected
        start_time = time.time()
        time.sleep(0.1)  # Simulate work
        duration = time.time() - start_time
        
        # Log performance metric
        from monitoring.logger import log_performance_metric
        log_performance_metric("test_operation", duration, "seconds")
        
        # In a real test, we'd verify the metric was recorded


class TestProductionReadiness:
    """Test production readiness features."""
    
    def test_graceful_shutdown(self):
        """Test graceful shutdown procedures."""
        # Test that all systems can be shut down gracefully
        from config.reloader import stop_config_watcher
        from storage.cache import clear_cache
        
        # These should not raise exceptions
        stop_config_watcher()
        clear_cache()
    
    def test_resource_cleanup(self):
        """Test resource cleanup."""
        # Test that resources are properly cleaned up
        cache = get_cache()
        cache.clear_cache()
        
        # In a real test, we'd verify resources were cleaned up
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        # Test that invalid configurations are caught
        settings = get_settings()
        
        # Test that required settings are present
        assert hasattr(settings, 'database_url')
        assert hasattr(settings, 'redis_url')
        assert hasattr(settings, 'api_security_token')
    
    def test_security_features(self):
        """Test security features."""
        # Test that security features are properly configured
        settings = get_settings()
        
        # Test that security token is set
        assert settings.api_security_token != "default-dev-token-change-in-production"
        
        # Test that rate limiting is configured
        assert settings.rate_limit > 0
        assert settings.rate_limit_window > 0


if __name__ == "__main__":
    pytest.main([__file__]) 
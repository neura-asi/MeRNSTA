#!/usr/bin/env python3
"""
MeRNSTA Performance Benchmarking Suite
Tests performance across different hardware configurations and system loads
"""

import sys
import os
import time
import psutil
import platform
import json
import tempfile
import statistics
from datetime import datetime
from typing import Dict, List, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.memory_log import MemoryLog
from cortex.memory_ops import process_user_input
from scripts.embedder import embed
import numpy as np

class PerformanceBenchmark:
    """Comprehensive performance benchmarking for MeRNSTA"""
    
    def __init__(self):
        self.results = {}
        self.system_info = self._get_system_info()
        
    def _get_system_info(self) -> Dict[str, Any]:
        """Get detailed system information"""
        cpu_info = {}
        try:
            # Try to get more detailed CPU info
            with open('/proc/cpuinfo', 'r') as f:
                lines = f.readlines()
            cpu_info['model'] = next((line.split(':')[1].strip() for line in lines if 'model name' in line), 'Unknown')
            cpu_info['cores'] = psutil.cpu_count(logical=False)
            cpu_info['threads'] = psutil.cpu_count(logical=True)
        except:
            cpu_info = {
                'model': platform.processor() or 'Unknown',
                'cores': psutil.cpu_count(logical=False),
                'threads': psutil.cpu_count(logical=True)
            }
        
        memory = psutil.virtual_memory()
        
        return {
            'platform': platform.platform(),
            'cpu': cpu_info,
            'memory_gb': round(memory.total / (1024**3), 1),
            'python_version': platform.python_version(),
            'timestamp': datetime.now().isoformat()
        }
    
    def benchmark_memory_operations(self, fact_counts: List[int] = [100, 1000, 5000, 10000]) -> Dict[str, Any]:
        """Benchmark memory operations at different scales"""
        print("🧪 Benchmarking Memory Operations")
        results = {}
        
        for count in fact_counts:
            print(f"   Testing with {count} facts...")
            
            # Create temporary database
            with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
                db_path = tmp.name
            
            try:
                memory_log = MemoryLog(db_path)
                
                # Benchmark fact storage
                storage_times = []
                test_facts = [
                    f"User likes item_{i % 100}" for i in range(count)
                ]
                
                start_time = time.time()
                for fact_text in test_facts:
                    fact_start = time.time()
                    triplets = memory_log.extract_triplets(fact_text)
                    if triplets:
                        memory_log.store_triplets(triplets)
                    storage_times.append((time.time() - fact_start) * 1000)  # ms
                
                total_storage_time = time.time() - start_time
                
                # Benchmark retrieval
                retrieval_times = []
                queries = [
                    "What does the user like?",
                    "Tell me about preferences",
                    "What are the user's interests?"
                ]
                
                for query in queries * 10:  # Run each query 10 times
                    retrieval_start = time.time()
                    facts = memory_log.get_all_facts()
                    retrieval_times.append((time.time() - retrieval_start) * 1000)
                
                # Benchmark semantic search
                search_times = []
                for query in queries * 5:
                    search_start = time.time()
                    results_search = memory_log.semantic_search(query, topk=5)
                    search_times.append((time.time() - search_start) * 1000)
                
                results[f"{count}_facts"] = {
                    'total_storage_time_ms': total_storage_time * 1000,
                    'avg_storage_time_ms': statistics.mean(storage_times),
                    'median_storage_time_ms': statistics.median(storage_times),
                    'p95_storage_time_ms': np.percentile(storage_times, 95),
                    'avg_retrieval_time_ms': statistics.mean(retrieval_times),
                    'median_retrieval_time_ms': statistics.median(retrieval_times),
                    'p95_retrieval_time_ms': np.percentile(retrieval_times, 95),
                    'avg_search_time_ms': statistics.mean(search_times),
                    'median_search_time_ms': statistics.median(search_times),
                    'p95_search_time_ms': np.percentile(search_times, 95),
                    'facts_stored': len(memory_log.get_all_facts())
                }
                
            finally:
                # Cleanup
                try:
                    os.unlink(db_path)
                except:
                    pass
        
        return results
    
    def benchmark_contradiction_detection(self, iterations: int = 100) -> Dict[str, Any]:
        """Benchmark contradiction detection performance"""
        print("🧪 Benchmarking Contradiction Detection")
        
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        try:
            memory_log = MemoryLog(db_path)
            
            # Add contradictory facts
            contradictory_pairs = [
                ("I love pizza", "I hate pizza"),
                ("I prefer cats", "I prefer dogs"),
                ("I like summer", "I like winter"),
                ("I enjoy coding", "I dislike coding"),
                ("I drink coffee", "I don't drink coffee")
            ]
            
            # Store initial facts
            for fact1, fact2 in contradictory_pairs:
                triplets1 = memory_log.extract_triplets(fact1)
                triplets2 = memory_log.extract_triplets(fact2)
                if triplets1:
                    memory_log.store_triplets(triplets1)
                if triplets2:
                    memory_log.store_triplets(triplets2)
            
            # Benchmark contradiction detection
            detection_times = []
            for i in range(iterations):
                start_time = time.time()
                contradictions = memory_log.get_contradictions()
                detection_times.append((time.time() - start_time) * 1000)
            
            return {
                'avg_detection_time_ms': statistics.mean(detection_times),
                'median_detection_time_ms': statistics.median(detection_times),
                'p95_detection_time_ms': np.percentile(detection_times, 95),
                'contradictions_found': len(contradictions) if 'contradictions' in locals() else 0,
                'iterations': iterations
            }
            
        finally:
            try:
                os.unlink(db_path)
            except:
                pass
    
    def benchmark_embedding_operations(self, text_sizes: List[int] = [10, 50, 100, 500]) -> Dict[str, Any]:
        """Benchmark embedding generation at different text sizes"""
        print("🧪 Benchmarking Embedding Operations")
        results = {}
        
        for size in text_sizes:
            # Generate text of specified word count
            text = " ".join([f"word{i}" for i in range(size)])
            
            embedding_times = []
            for _ in range(10):  # 10 iterations per size
                try:
                    start_time = time.time()
                    embedding = embed(text)
                    embedding_times.append((time.time() - start_time) * 1000)
                except Exception as e:
                    print(f"   Warning: Embedding failed for {size} words: {e}")
                    continue
            
            if embedding_times:
                results[f"{size}_words"] = {
                    'avg_time_ms': statistics.mean(embedding_times),
                    'median_time_ms': statistics.median(embedding_times),
                    'p95_time_ms': np.percentile(embedding_times, 95),
                    'words_per_ms': size / statistics.mean(embedding_times) if embedding_times else 0
                }
        
        return results
    
    def benchmark_concurrent_load(self, concurrent_users: List[int] = [1, 5, 10, 20]) -> Dict[str, Any]:
        """Benchmark system under concurrent load"""
        print("🧪 Benchmarking Concurrent Load")
        results = {}
        
        def simulate_user_session(user_id: int, num_operations: int = 10) -> List[float]:
            """Simulate a user session with multiple operations"""
            times = []
            
            with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
                db_path = tmp.name
            
            try:
                memory_log = MemoryLog(db_path)
                
                for i in range(num_operations):
                    start_time = time.time()
                    
                    # Simulate typical user operations
                    if i % 3 == 0:
                        # Store a fact
                        text = f"User {user_id} likes activity {i}"
                        triplets = memory_log.extract_triplets(text)
                        if triplets:
                            memory_log.store_triplets(triplets)
                    elif i % 3 == 1:
                        # Query facts
                        facts = memory_log.get_all_facts()
                    else:
                        # Semantic search
                        memory_log.semantic_search(f"user {user_id} preferences", topk=3)
                    
                    times.append((time.time() - start_time) * 1000)
                
                return times
                
            finally:
                try:
                    os.unlink(db_path)
                except:
                    pass
        
        for num_users in concurrent_users:
            print(f"   Testing with {num_users} concurrent users...")
            
            overall_start = time.time()
            
            with ThreadPoolExecutor(max_workers=num_users) as executor:
                futures = [
                    executor.submit(simulate_user_session, user_id, 5)
                    for user_id in range(num_users)
                ]
                
                all_times = []
                for future in as_completed(futures):
                    try:
                        user_times = future.result()
                        all_times.extend(user_times)
                    except Exception as e:
                        print(f"   User session failed: {e}")
            
            overall_time = time.time() - overall_start
            
            if all_times:
                results[f"{num_users}_users"] = {
                    'total_time_s': overall_time,
                    'avg_operation_time_ms': statistics.mean(all_times),
                    'median_operation_time_ms': statistics.median(all_times),
                    'p95_operation_time_ms': np.percentile(all_times, 95),
                    'operations_per_second': len(all_times) / overall_time,
                    'total_operations': len(all_times)
                }
        
        return results
    
    def run_full_benchmark(self) -> Dict[str, Any]:
        """Run the complete benchmark suite"""
        print("🚀 Starting MeRNSTA Performance Benchmark Suite")
        print(f"System: {self.system_info['platform']}")
        print(f"CPU: {self.system_info['cpu']['model']} ({self.system_info['cpu']['cores']} cores)")
        print(f"Memory: {self.system_info['memory_gb']} GB")
        print("=" * 80)
        
        benchmark_start = time.time()
        
        # Run all benchmarks
        try:
            self.results['memory_operations'] = self.benchmark_memory_operations()
            self.results['contradiction_detection'] = self.benchmark_contradiction_detection()
            self.results['embedding_operations'] = self.benchmark_embedding_operations()
            self.results['concurrent_load'] = self.benchmark_concurrent_load()
        except Exception as e:
            print(f"Benchmark error: {e}")
        
        total_time = time.time() - benchmark_start
        
        # Compile final results
        final_results = {
            'system_info': self.system_info,
            'benchmark_duration_s': total_time,
            'results': self.results,
            'summary': self._generate_summary()
        }
        
        return final_results
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate performance summary"""
        summary = {}
        
        try:
            # Memory operations summary
            if 'memory_operations' in self.results:
                ops = self.results['memory_operations']
                if '1000_facts' in ops:
                    summary['storage_latency_1k_facts_ms'] = ops['1000_facts']['avg_storage_time_ms']
                    summary['retrieval_latency_1k_facts_ms'] = ops['1000_facts']['avg_retrieval_time_ms']
                    summary['search_latency_1k_facts_ms'] = ops['1000_facts']['avg_search_time_ms']
            
            # Contradiction detection summary
            if 'contradiction_detection' in self.results:
                summary['contradiction_detection_ms'] = self.results['contradiction_detection']['avg_detection_time_ms']
            
            # Embedding operations summary
            if 'embedding_operations' in self.results:
                ops = self.results['embedding_operations']
                if '50_words' in ops:
                    summary['embedding_50_words_ms'] = ops['50_words']['avg_time_ms']
            
            # Concurrent load summary
            if 'concurrent_load' in self.results:
                load = self.results['concurrent_load']
                if '10_users' in load:
                    summary['ops_per_second_10_users'] = load['10_users']['operations_per_second']
        
        except Exception as e:
            summary['error'] = str(e)
        
        return summary
    
    def save_results(self, filename: str = None):
        """Save benchmark results to JSON file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump({
                'system_info': self.system_info,
                'results': self.results
            }, f, indent=2)
        
        print(f"\n📊 Results saved to: {filename}")
        return filename
    
    def print_summary(self):
        """Print human-readable benchmark summary"""
        print("\n" + "=" * 80)
        print("📊 PERFORMANCE BENCHMARK SUMMARY")
        print("=" * 80)
        
        print(f"🖥️  System: {self.system_info['cpu']['model']}")
        print(f"💾 Memory: {self.system_info['memory_gb']} GB")
        print(f"🔧 Cores: {self.system_info['cpu']['cores']} physical, {self.system_info['cpu']['threads']} logical")
        
        summary = self._generate_summary()
        
        if 'storage_latency_1k_facts_ms' in summary:
            print(f"\n⚡ Performance Metrics (1,000 facts):")
            print(f"   Storage: {summary['storage_latency_1k_facts_ms']:.2f}ms per fact")
            print(f"   Retrieval: {summary['retrieval_latency_1k_facts_ms']:.2f}ms")
            print(f"   Search: {summary['search_latency_1k_facts_ms']:.2f}ms")
        
        if 'contradiction_detection_ms' in summary:
            print(f"   Contradiction Detection: {summary['contradiction_detection_ms']:.2f}ms")
        
        if 'embedding_50_words_ms' in summary:
            print(f"   Embedding (50 words): {summary['embedding_50_words_ms']:.2f}ms")
        
        if 'ops_per_second_10_users' in summary:
            print(f"   Throughput (10 users): {summary['ops_per_second_10_users']:.1f} ops/sec")
        
        print("\n💡 Hardware Impact:")
        print("   - CPU: Affects embedding generation and text processing")
        print("   - Memory: Affects cache performance and concurrent operations")
        print("   - Storage: Affects database I/O performance")
        print("   - Network: Affects Ollama API calls (if remote)")

def main():
    """Run benchmark suite"""
    benchmark = PerformanceBenchmark()
    results = benchmark.run_full_benchmark()
    benchmark.print_summary()
    benchmark.save_results()

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Performance Monitor for Financial Graph RAG System

This script helps monitor and optimize the performance of the Streamlit application
by tracking initialization times, cache performance, and system bottlenecks.
"""

import time
import logging
import psutil
import os
from typing import Dict, Any
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitor system performance and identify bottlenecks"""
    
    def __init__(self):
        self.start_time = time.time()
        self.metrics = {}
        
    def start_timer(self, operation: str):
        """Start timing an operation"""
        self.metrics[operation] = {
            'start_time': time.time(),
            'memory_start': psutil.Process().memory_info().rss / 1024 / 1024  # MB
        }
        logger.info(f"Starting {operation}")
        
    def end_timer(self, operation: str) -> Dict[str, Any]:
        """End timing an operation and return metrics"""
        if operation not in self.metrics:
            logger.warning(f"No start time found for {operation}")
            return {}
            
        end_time = time.time()
        memory_end = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        duration = end_time - self.metrics[operation]['start_time']
        memory_used = memory_end - self.metrics[operation]['memory_start']
        
        metrics = {
            'duration': duration,
            'memory_used': memory_used,
            'memory_total': memory_end
        }
        
        logger.info(f"Completed {operation} in {duration:.2f}s (Memory: {memory_used:.1f}MB)")
        
        # Clean up
        del self.metrics[operation]
        
        return metrics
        
    def get_system_info(self) -> Dict[str, Any]:
        """Get current system information"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_available': psutil.virtual_memory().available / 1024 / 1024 / 1024,  # GB
            'disk_usage': psutil.disk_usage('/').percent
        }
        
    def check_cache_performance(self, cache_dir: str = "cache") -> Dict[str, Any]:
        """Check cache directory performance"""
        cache_path = Path(cache_dir)
        if not cache_path.exists():
            return {'error': 'Cache directory not found'}
            
        cache_files = list(cache_path.glob('*'))
        total_size = sum(f.stat().st_size for f in cache_files if f.is_file())
        
        return {
            'cache_files': len(cache_files),
            'cache_size_mb': total_size / 1024 / 1024,
            'cache_path': str(cache_path.absolute())
        }
        
    def monitor_initialization(self):
        """Monitor system initialization performance"""
        logger.info("Starting system initialization monitoring")
        
        try:
            # Monitor system import
            self.start_timer("system_import")
            from financial_graph_rag.core import FinancialVectorRAG
            import_metrics = self.end_timer("system_import")
            
            # Monitor system initialization
            self.start_timer("system_init")
            system = FinancialVectorRAG()
            init_metrics = self.end_timer("system_init")
            
            # Monitor component initialization
            self.start_timer("rag_engine_init")
            rag_engine = system.rag_engine
            rag_metrics = self.end_timer("rag_engine_init")
            
            # Monitor skill shortage analyzer
            self.start_timer("skill_analyzer_init")
            skill_analyzer = system.skill_shortage_analyzer
            skill_metrics = self.end_timer("skill_analyzer_init")
            
            # Get system info
            system_info = self.get_system_info()
            cache_info = self.check_cache_performance()
            
            # Compile report
            report = {
                'import_time': import_metrics.get('duration', 0),
                'init_time': init_metrics.get('duration', 0),
                'rag_init_time': rag_metrics.get('duration', 0),
                'skill_init_time': skill_metrics.get('duration', 0),
                'total_init_time': import_metrics.get('duration', 0) + init_metrics.get('duration', 0),
                'memory_usage': init_metrics.get('memory_total', 0),
                'system_info': system_info,
                'cache_info': cache_info
            }
            
            logger.info("Performance monitoring completed")
            return report
            
        except Exception as e:
            logger.error(f"Error during performance monitoring: {e}")
            return {'error': str(e)}
    
    def generate_performance_report(self) -> str:
        """Generate a formatted performance report"""
        report = self.monitor_initialization()
        
        if 'error' in report:
            return f"Error: {report['error']}"
            
        report_text = f"""
=== Financial Graph RAG Performance Report ===

Initialization Times:
- Import time: {report['import_time']:.2f}s
- System init: {report['init_time']:.2f}s
- RAG engine init: {report['rag_init_time']:.2f}s
- Skill analyzer init: {report['skill_init_time']:.2f}s
- Total init time: {report['total_init_time']:.2f}s

Memory Usage:
- Total memory: {report['memory_usage']:.1f}MB

System Resources:
- CPU usage: {report['system_info']['cpu_percent']:.1f}%
- Memory usage: {report['system_info']['memory_percent']:.1f}%
- Available memory: {report['system_info']['memory_available']:.1f}GB
- Disk usage: {report['system_info']['disk_usage']:.1f}%

Cache Performance:
- Cache files: {report['cache_info'].get('cache_files', 0)}
- Cache size: {report['cache_info'].get('cache_size_mb', 0):.1f}MB
- Cache path: {report['cache_info'].get('cache_path', 'N/A')}

Performance Recommendations:
"""
        
        # Add recommendations based on metrics
        if report['total_init_time'] > 30:
            report_text += "- ⚠️  Initialization is slow (>30s). Consider pre-loading models.\n"
        else:
            report_text += "- ✅ Initialization time is acceptable.\n"
            
        if report['memory_usage'] > 2000:  # 2GB
            report_text += "- ⚠️  High memory usage. Consider model optimization.\n"
        else:
            report_text += "- ✅ Memory usage is reasonable.\n"
            
        if report['system_info']['memory_percent'] > 80:
            report_text += "- ⚠️  System memory usage is high. Close other applications.\n"
        else:
            report_text += "- ✅ System memory usage is good.\n"
            
        return report_text

def main():
    """Main function to run performance monitoring"""
    print("Starting Financial Graph RAG Performance Monitor...")
    
    monitor = PerformanceMonitor()
    report = monitor.generate_performance_report()
    
    print(report)
    
    # Save report to file
    with open('performance_report.txt', 'w') as f:
        f.write(report)
    
    print("\nPerformance report saved to 'performance_report.txt'")

if __name__ == "__main__":
    main() 
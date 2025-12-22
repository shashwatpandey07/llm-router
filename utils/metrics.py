"""
Metrics logging utilities.

Provides CSV logging for routing decisions, costs, and performance metrics.
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


class MetricsLogger:
    """
    Logs routing metrics to CSV and JSON files.
    
    Tracks:
    - Query text and difficulty
    - Routing decisions
    - Token usage
    - Latency
    - Cost (USD)
    - Cost savings
    """
    
    def __init__(self, log_dir: str = "logs"):
        """
        Initialize metrics logger.
        
        Args:
            log_dir: Directory to store log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # CSV file for structured logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_file = self.log_dir / f"routing_metrics_{timestamp}.csv"
        self.json_file = self.log_dir / f"routing_metrics_{timestamp}.json"
        
        # Initialize CSV with headers
        self._init_csv()
        
        # Store all metrics for JSON export
        self.metrics = []
    
    def _init_csv(self):
        """Initialize CSV file with headers."""
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "query",
                "difficulty",
                "routing_decision",
                "model",
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "latency_ms",
                "cost_usd",
                "cost_saved_usd",
                "device"
            ])
    
    def log(self, result: Dict, query: str):
        """
        Log a routing result.
        
        Args:
            result: Result dictionary from router.route()
            query: Original query string
        """
        timestamp = datetime.now().isoformat()
        
        # Prepare CSV row
        row = [
            timestamp,
            query[:200],  # Truncate long queries
            result.get("difficulty", 0.0),
            result.get("routing_decision", "unknown"),
            result.get("model", "unknown"),
            result.get("input_tokens", 0),
            result.get("output_tokens", 0),
            result.get("input_tokens", 0) + result.get("output_tokens", 0),
            result.get("latency_ms", 0.0),
            result.get("cost_usd", 0.0),
            result.get("cost_saved_usd", 0.0),
            result.get("device", "unknown")
        ]
        
        # Write to CSV
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        
        # Store for JSON export
        self.metrics.append({
            "timestamp": timestamp,
            "query": query,
            **result
        })
    
    def export_json(self):
        """Export all metrics to JSON file."""
        with open(self.json_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def get_summary(self) -> Dict:
        """
        Get summary statistics from logged metrics.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.metrics:
            return {}
        
        total_queries = len(self.metrics)
        routing_decisions = {}
        total_cost = 0.0
        total_saved = 0.0
        total_latency = 0.0
        total_tokens = 0
        
        for metric in self.metrics:
            decision = metric.get("routing_decision", "unknown")
            routing_decisions[decision] = routing_decisions.get(decision, 0) + 1
            total_cost += metric.get("cost_usd", 0.0)
            total_saved += metric.get("cost_saved_usd", 0.0)
            total_latency += metric.get("latency_ms", 0.0)
            total_tokens += metric.get("input_tokens", 0) + metric.get("output_tokens", 0)
        
        return {
            "total_queries": total_queries,
            "routing_decisions": routing_decisions,
            "total_cost_usd": round(total_cost, 6),
            "total_cost_saved_usd": round(total_saved, 6),
            "avg_latency_ms": round(total_latency / total_queries, 2) if total_queries > 0 else 0,
            "total_tokens": total_tokens,
            "csv_file": str(self.csv_file),
            "json_file": str(self.json_file)
        }


"""JSON-based result storage -- no database required."""

import json
import os
from datetime import datetime


class ResultsStore:
    """Store and retrieve benchmark results as JSON files."""

    def __init__(self, data_dir: str):
        """Initialize the results store.

        Args:
            data_dir: Directory to store JSON result files.
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def save_result(self, run_id: str, results: dict) -> str:
        """Save benchmark results to a JSON file.

        Args:
            run_id: Unique identifier for this run.
            results: Results dictionary.

        Returns:
            Path to the saved file.
        """
        filepath = os.path.join(self.data_dir, f"{run_id}.json")
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)
        return filepath

    def load_result(self, run_id: str) -> dict:
        """Load benchmark results from a JSON file.

        Args:
            run_id: Run identifier.

        Returns:
            Results dictionary.
        """
        filepath = os.path.join(self.data_dir, f"{run_id}.json")
        with open(filepath) as f:
            return json.load(f)

    def list_runs(self) -> list[dict]:
        """List all saved benchmark runs with metadata.

        Returns:
            List of run metadata dictionaries.
        """
        runs = []
        for fname in sorted(os.listdir(self.data_dir)):
            if fname.endswith(".json"):
                run_id = fname[:-5]
                try:
                    data = self.load_result(run_id)
                    runs.append(
                        {
                            "run_id": run_id,
                            "timestamp": data.get("timestamp", ""),
                            "mode": data.get("mode", "unknown"),
                            "model": data.get("model", "unknown"),
                            "total": data.get("total", 0),
                            "agent_accuracy": data.get("agent", {}).get("accuracy", 0),
                            "raw_accuracy": data.get("raw_llm", {}).get("accuracy", 0),
                        }
                    )
                except Exception:
                    continue
        return runs

    def delete_result(self, run_id: str) -> None:
        """Delete a saved result.

        Args:
            run_id: Run identifier to delete.
        """
        filepath = os.path.join(self.data_dir, f"{run_id}.json")
        if os.path.exists(filepath):
            os.remove(filepath)

    def get_comparison_data(self) -> list[dict]:
        """Get data for comparison across runs.

        Returns:
            List of run metadata.
        """
        return self.list_runs()

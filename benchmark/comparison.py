"""Compare agent vs raw LLM benchmark results."""


class ComparisonAnalyzer:
    """Analyze and compare agent vs raw LLM performance on benchmarks."""

    @staticmethod
    def compare_runs(agent_results: dict, raw_results: dict) -> dict:
        """Compare agent and raw LLM benchmark results.

        Args:
            agent_results: Agent results dictionary with 'accuracy' and 'results' keys.
            raw_results: Raw LLM results dictionary with 'accuracy' and 'results' keys.

        Returns:
            Comparison dictionary with breakdown statistics.
        """
        agent_acc = agent_results.get("accuracy", 0)
        raw_acc = raw_results.get("accuracy", 0)

        a_items = {i: r for i, r in enumerate(agent_results.get("results", []))}
        r_items = {i: r for i, r in enumerate(raw_results.get("results", []))}

        agent_wins = 0
        raw_wins = 0
        both_correct = 0
        both_wrong = 0
        per_problem = []

        for i in sorted(set(list(a_items.keys()) + list(r_items.keys()))):
            a_correct = a_items.get(i, {}).get("correct", False)
            r_correct = r_items.get(i, {}).get("correct", False)

            if a_correct and not r_correct:
                agent_wins += 1
            elif r_correct and not a_correct:
                raw_wins += 1
            elif a_correct and r_correct:
                both_correct += 1
            else:
                both_wrong += 1

            problem_text = a_items.get(i, {}).get(
                "problem", r_items.get(i, {}).get("problem", "")
            )
            per_problem.append(
                {
                    "index": i,
                    "problem": problem_text,
                    "agent_correct": a_correct,
                    "raw_correct": r_correct,
                    "agent_answer": a_items.get(i, {}).get("predicted", ""),
                    "raw_answer": r_items.get(i, {}).get("predicted", ""),
                    "gold_answer": a_items.get(i, {}).get(
                        "gold_answer", r_items.get(i, {}).get("gold_answer", "")
                    ),
                }
            )

        return {
            "agent_accuracy": agent_acc,
            "raw_accuracy": raw_acc,
            "accuracy_diff": round(agent_acc - raw_acc, 4),
            "agent_avg_time": agent_results.get("avg_time", 0),
            "raw_avg_time": raw_results.get("avg_time", 0),
            "agent_wins": agent_wins,
            "raw_wins": raw_wins,
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "per_problem": per_problem,
        }

    @staticmethod
    def generate_report(comparison: dict) -> str:
        """Generate a markdown comparison report.

        Args:
            comparison: Comparison dictionary from compare_runs().

        Returns:
            Markdown formatted report string.
        """
        lines = [
            "# Benchmark Comparison Report",
            "",
            f"**Agent Accuracy:** {comparison['agent_accuracy']:.1%}",
            f"**Raw LLM Accuracy:** {comparison['raw_accuracy']:.1%}",
            f"**Accuracy Difference:** {comparison['accuracy_diff']:+.1%}",
            "",
            "## Problem Breakdown",
            "",
            f"- Both correct: {comparison['both_correct']}",
            f"- Agent wins: {comparison['agent_wins']} (agent got right, LLM got wrong)",
            f"- Raw LLM wins: {comparison['raw_wins']}",
            f"- Both wrong: {comparison['both_wrong']}",
            "",
        ]
        return "\n".join(lines)

    @staticmethod
    def get_winning_problems(comparison: dict) -> list[dict]:
        """Get problems where the agent won (correct) and raw LLM lost (wrong).

        Args:
            comparison: Comparison dictionary from compare_runs().

        Returns:
            List of problem dictionaries sorted by problem length descending.
        """
        wins = [
            p
            for p in comparison["per_problem"]
            if p["agent_correct"] and not p["raw_correct"]
        ]
        return sorted(wins, key=lambda x: len(x["problem"]), reverse=True)

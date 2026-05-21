#!/usr/bin/env python3
"""CLI entry point for the JEE Advanced Math Reasoning Agent."""

import argparse
import os
import sys


def get_config_from_env():
    """Build a Config object from environment variables."""
    from config import Config

    return Config()


def print_banner():
    """Print the CLI banner."""
    print("=" * 60)
    print("  JEE Advanced Math Reasoning Agent - CLI")
    print("  LangGraph + Code Execution + SymPy Verification")
    print("=" * 60)


def cmd_solve(problem, cfg):
    """Solve a single problem using the agent."""
    from agent.graph import MathAgent

    print(f"\n Problem: {problem}")
    print("-" * 60)

    try:
        cfg.validate()
        agent = MathAgent(cfg)
        result = agent.solve(problem)

        print(f"\n Final Answer: {result.get('final_answer', 'N/A')}")
        print(f" Retries: {result.get('retry_count', 0)}")
        print(f" Reasoning steps: {len(result.get('reasoning_trace', []))}")

        print("\n Reasoning Trace:")
        for i, entry in enumerate(result.get("reasoning_trace", []), 1):
            print(f"  Step {i}:")
            for line in entry.split("\n"):
                print(f"    {line}")

        if result.get("code"):
            print(f"\n Code Executed:")
            for line in result["code"].split("\n"):
                print(f"    {line}")

        if result.get("code_result"):
            print(f"\n Code Output: {result['code_result']}")

        if result.get("verification"):
            print(f"\n Verification: {result['verification']}")

    except Exception as exc:
        print(f"\n Error: {exc}", file=sys.stderr)
        sys.exit(1)


def cmd_compare(cfg):
    """Compare the latest two benchmark runs."""
    from benchmark.comparison import ComparisonAnalyzer
    from benchmark.results import ResultsStore

    try:
        store = ResultsStore(cfg.DATA_DIR)
        runs = store.list_runs()

        if len(runs) < 2:
            print("Need at least 2 saved runs to compare.", file=sys.stderr)
            sys.exit(1)

        # Get the two most recent runs
        recent = runs[-2:]
        run1 = store.load_result(recent[0]["run_id"])
        run2 = store.load_result(recent[1]["run_id"])

        if "agent" in run1 and "raw_llm" in run1:
            comparison = ComparisonAnalyzer.compare_runs(
                run1["agent"], run1["raw_llm"]
            )
            print("\n" + ComparisonAnalyzer.generate_report(comparison))
        elif "agent" in run2 and "raw_llm" in run2:
            comparison = ComparisonAnalyzer.compare_runs(
                run2["agent"], run2["raw_llm"]
            )
            print("\n" + ComparisonAnalyzer.generate_report(comparison))
        else:
            print("No run with both agent and raw LLM results found.", file=sys.stderr)

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


def cmd_jee_benchmark(count, mode, problem_type, difficulty, cfg):
    """Run the JEE Advanced benchmark."""
    from agent.graph import MathAgent
    from benchmark.jee_evaluator import JEEEValuator
    from benchmark.jee_loader import JEELoader

    print(f"\n Running JEE Advanced Benchmark")
    print(f"  Problems: {count}")
    print(f"  Mode: {mode}")
    print(f"  Model: {cfg.LLM_MODEL}")
    if problem_type:
        print(f"  Problem Type: {problem_type}")
    if difficulty:
        print(f"  Difficulty: {difficulty}")
    print("-" * 60)

    try:
        cfg.validate()

        # Load dataset with filters
        dataset = JEELoader.load_problems(
            max_problems=count,
            problem_type=problem_type,
            difficulty=difficulty,
        )
        print(f"  Loaded {len(dataset)} JEE problems")

        if len(dataset) == 0:
            print("  No problems match the given filters.", file=sys.stderr)
            sys.exit(1)

        # Initialize agent and evaluator
        agent = MathAgent(cfg)
        evaluator = JEEEValuator(agent, cfg)

        # Progress callback
        def progress(current, total):
            pct = int(100 * current / total)
            print(f"  Progress: [{current}/{total}] {pct}%", end="\r")

        # Run benchmark
        results = evaluator.run_benchmark(
            dataset, mode=mode, progress_callback=progress
        )
        print()  # newline after progress

        # Print results
        print("\n Results:")
        if "agent" in results:
            a = results["agent"]
            print(f"  Agent:  {a['correct']}/{results['total']} correct ({a['accuracy']:.1%})")
            print(f"  Avg time: {a['avg_time']}s")

            # Per-type breakdown
            if a.get("by_type"):
                print("\n  By Problem Type:")
                for t, d in a["by_type"].items():
                    print(f"    {t}: {d['correct']}/{d['total']} ({d['accuracy']:.0%})")

            # Per-difficulty breakdown
            if a.get("by_difficulty"):
                print("\n  By Difficulty:")
                for diff, d in a["by_difficulty"].items():
                    print(f"    {diff}: {d['correct']}/{d['total']} ({d['accuracy']:.0%})")

        if "raw_llm" in results:
            r = results["raw_llm"]
            print(f"\n  Raw LLM: {r['correct']}/{results['total']} correct ({r['accuracy']:.1%})")
            print(f"  Avg time: {r['avg_time']}s")

        if "agent" in results and "raw_llm" in results:
            a_acc = results["agent"]["accuracy"]
            r_acc = results["raw_llm"]["accuracy"]
            diff = a_acc - r_acc
            if diff > 0:
                print(f"\n  Agent outperforms raw LLM by {diff:.1%}")
            elif diff < 0:
                print(f"\n  Raw LLM outperforms agent by {abs(diff):.1%}")
            else:
                print(f"\n  Agent and raw LLM tied at {a_acc:.1%}")

        # Save results
        from benchmark.results import ResultsStore

        store = ResultsStore(cfg.DATA_DIR)
        filepath = store.save_result(results["run_id"], results)
        print(f"\n  Results saved to: {filepath}")

    except Exception as exc:
        print(f"\n Error: {exc}", file=sys.stderr)
        sys.exit(1)


def cmd_dashboard():
    """Launch the Streamlit dashboard."""
    import subprocess

    script_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(script_dir, "interface", "jee_app.py")

    print("\n Launching JEE Streamlit dashboard...")
    print(f"  App: {app_path}")
    print("-" * 60)

    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", app_path],
        cwd=script_dir,
    )


def main():
    """Parse CLI arguments and execute commands."""
    parser = argparse.ArgumentParser(
        description="JEE Advanced Math Reasoning Agent CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --problem "Evaluate lim x->0 sin(3x)/x"
  python run.py --jee --count 10 --mode both
  python run.py --jee --jee-type limits --jee-difficulty medium
  python run.py --dashboard
  python run.py --compare
        """,
    )

    parser.add_argument(
        "--problem", "-p", type=str, help="Solve a single math problem"
    )
    parser.add_argument(
        "--jee", action="store_true", help="Run JEE Advanced benchmark"
    )
    parser.add_argument(
        "--count", "-c", type=int, default=20, help="Number of problems (default: 20)"
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["agent", "raw_llm", "both"],
        default="both",
        help="Benchmark mode (default: both)",
    )
    parser.add_argument(
        "--jee-type", type=str, default=None,
        help="Filter by problem type (limits, differentiation, integration, definite_integrals, differential_equations, maxima_minima, tangent_normal, area)"
    )
    parser.add_argument(
        "--jee-difficulty", type=str, default=None,
        choices=["easy", "medium", "hard"],
        help="Filter by difficulty level"
    )
    parser.add_argument(
        "--dashboard", "-d", action="store_true", help="Launch Streamlit dashboard"
    )
    parser.add_argument(
        "--compare", action="store_true", help="Compare latest two benchmark runs"
    )

    args = parser.parse_args()

    print_banner()

    if args.dashboard:
        cmd_dashboard()
    elif args.jee:
        cfg = get_config_from_env()
        cmd_jee_benchmark(args.count, args.mode, args.jee_type, args.jee_difficulty, cfg)
    elif args.problem:
        cfg = get_config_from_env()
        cmd_solve(args.problem, cfg)
    elif args.compare:
        cfg = get_config_from_env()
        cmd_compare(cfg)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

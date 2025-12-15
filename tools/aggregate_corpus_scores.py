#!/usr/bin/env python3
"""
Aggregate corpus-wide semantic loss scores and generate report.

Collects all loss metrics from individual file comparisons and generates
comprehensive statistics, rankings, and markdown report.

Usage:
    ./aggregate_corpus_scores.py --scores-dir corpus/scores --output CORPUS_LOSS_REPORT.md
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass
from collections import defaultdict
import statistics


@dataclass
class FileScore:
    """Loss metrics for a single file."""
    file: str
    combined_loss: float
    structural: float
    type_dist: float
    operation: float
    matched: int
    missing: int
    extra: int


class CorpusAggregator:
    """Aggregates corpus-wide loss scores."""

    def __init__(self):
        self.scores: List[FileScore] = []

    def add_score_file(self, score_file: Path):
        """Add a loss score file to the aggregation."""
        try:
            data = json.loads(score_file.read_text())
            metrics = data.get("metrics", {})

            self.scores.append(FileScore(
                file=metrics.get("file", "unknown"),
                combined_loss=metrics.get("combined_loss", 0.0),
                structural=metrics.get("structural_distance", 0.0),
                type_dist=metrics.get("type_distance", 0.0),
                operation=metrics.get("operation_distance", 0.0),
                matched=metrics.get("matched_patterns", 0),
                missing=metrics.get("missing_patterns", 0),
                extra=metrics.get("extra_patterns", 0)
            ))
        except Exception as e:
            print(f"Warning: Failed to read {score_file.name}: {e}")

    def generate_report(self) -> str:
        """Generate markdown report with comprehensive statistics."""
        if not self.scores:
            return "# No scores to aggregate\n"

        # Compute statistics
        losses = [s.combined_loss for s in self.scores]
        avg_loss = statistics.mean(losses)
        median_loss = statistics.median(losses)
        min_loss = min(losses)
        max_loss = max(losses)

        zero_loss = sum(1 for s in self.scores if s.combined_loss == 0.0)
        low_loss = sum(1 for s in self.scores if 0.0 < s.combined_loss < 0.1)
        medium_loss = sum(1 for s in self.scores if 0.1 <= s.combined_loss < 0.5)
        high_loss = sum(1 for s in self.scores if s.combined_loss >= 0.5)

        total = len(self.scores)

        # Generate markdown
        lines = []
        lines.append("# Semantic Loss Report: cppfort vs cppfront\n")
        lines.append(f"**Total files**: {total}")
        lines.append(f"**Average loss**: {avg_loss:.3f}")
        lines.append(f"**Median loss**: {median_loss:.3f}")
        lines.append(f"**Min loss**: {min_loss:.3f}")
        lines.append(f"**Max loss**: {max_loss:.3f}")
        lines.append(f"**Zero-loss files**: {zero_loss} ({zero_loss/total*100:.1f}%)\n")

        lines.append("## Loss Distribution\n")
        lines.append("| Range | Count | Percentage |")
        lines.append("|-------|-------|------------|")
        lines.append(f"| 0.0 (perfect) | {zero_loss} | {zero_loss/total*100:.1f}% |")
        lines.append(f"| 0.0-0.1 (low) | {low_loss} | {low_loss/total*100:.1f}% |")
        lines.append(f"| 0.1-0.5 (medium) | {medium_loss} | {medium_loss/total*100:.1f}% |")
        lines.append(f"| >0.5 (high) | {high_loss} | {high_loss/total*100:.1f}% |\n")

        # High-loss files
        high_loss_files = sorted(
            [s for s in self.scores if s.combined_loss >= 0.5],
            key=lambda s: s.combined_loss,
            reverse=True
        )

        if high_loss_files:
            lines.append("## High-Loss Files (≥0.5)\n")
            lines.append("| File | Loss | Structural | Type | Operation | Missing | Extra |")
            lines.append("|------|------|------------|------|-----------|---------|-------|")
            for s in high_loss_files[:20]:  # Top 20
                lines.append(
                    f"| {s.file} | {s.combined_loss:.3f} | {s.structural:.3f} | "
                    f"{s.type_dist:.3f} | {s.operation:.3f} | {s.missing} | {s.extra} |"
                )
            lines.append("")

        # Perfect match files
        perfect_files = sorted([s for s in self.scores if s.combined_loss == 0.0])
        if perfect_files:
            lines.append(f"## Zero-Loss Files ({len(perfect_files)} files)\n")
            lines.append("| File | Matched Patterns |")
            lines.append("|------|------------------|")
            for s in perfect_files[:20]:  # Top 20
                lines.append(f"| {s.file} | {s.matched} |")
            if len(perfect_files) > 20:
                lines.append(f"| ... and {len(perfect_files) - 20} more | |")
            lines.append("")

        # Component breakdown
        lines.append("## Loss Component Breakdown\n")
        avg_structural = statistics.mean([s.structural for s in self.scores])
        avg_type = statistics.mean([s.type_dist for s in self.scores])
        avg_operation = statistics.mean([s.operation for s in self.scores])

        lines.append("| Component | Average | Contribution |")
        lines.append("|-----------|---------|--------------|")
        lines.append(f"| Structural | {avg_structural:.3f} | {avg_structural/avg_loss*100:.1f}% |")
        lines.append(f"| Type | {avg_type:.3f} | {avg_type/avg_loss*100:.1f}% |")
        lines.append(f"| Operation | {avg_operation:.3f} | {avg_operation/avg_loss*100:.1f}% |\n")

        # Pattern matching statistics
        total_matched = sum(s.matched for s in self.scores)
        total_missing = sum(s.missing for s in self.scores)
        total_extra = sum(s.extra for s in self.scores)

        lines.append("## Pattern Matching Statistics\n")
        lines.append(f"**Total matched patterns**: {total_matched}")
        lines.append(f"**Total missing patterns**: {total_missing}")
        lines.append(f"**Total extra patterns**: {total_extra}")
        lines.append(f"**Match rate**: {total_matched/(total_matched+total_missing)*100:.1f}%\n")

        # Worst performers by component
        lines.append("## Worst Performers by Component\n")

        worst_structural = sorted(self.scores, key=lambda s: s.structural, reverse=True)[:5]
        lines.append("### Highest Structural Distance\n")
        for s in worst_structural:
            lines.append(f"- {s.file}: {s.structural:.3f}")
        lines.append("")

        worst_type = sorted(self.scores, key=lambda s: s.type_dist, reverse=True)[:5]
        lines.append("### Highest Type Distance\n")
        for s in worst_type:
            lines.append(f"- {s.file}: {s.type_dist:.3f}")
        lines.append("")

        worst_operation = sorted(self.scores, key=lambda s: s.operation, reverse=True)[:5]
        lines.append("### Highest Operation Distance\n")
        for s in worst_operation:
            lines.append(f"- {s.file}: {s.operation:.3f}")
        lines.append("")

        # Footer
        lines.append("---\n")
        lines.append("**Generated by**: `aggregate_corpus_scores.py`")
        lines.append(f"**Files analyzed**: {total}")
        lines.append(f"**Overall quality**: ", end="")
        if avg_loss < 0.1:
            lines.append("**EXCELLENT** (avg loss < 0.1)")
        elif avg_loss < 0.3:
            lines.append("**GOOD** (avg loss < 0.3)")
        elif avg_loss < 0.5:
            lines.append("**FAIR** (avg loss < 0.5)")
        else:
            lines.append("**NEEDS WORK** (avg loss ≥ 0.5)")

        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate corpus-wide semantic loss scores"
    )
    parser.add_argument(
        "--scores-dir",
        type=Path,
        required=True,
        help="Directory containing loss score JSON files"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output markdown report file"
    )

    args = parser.parse_args()

    # Aggregate scores
    print(f"Scanning scores in: {args.scores_dir}")
    aggregator = CorpusAggregator()

    score_files = sorted(args.scores_dir.glob("*.loss.json"))
    print(f"Found {len(score_files)} score files")

    for score_file in score_files:
        aggregator.add_score_file(score_file)

    print(f"Aggregated {len(aggregator.scores)} scores")

    # Generate report
    print("Generating report...")
    report = aggregator.generate_report()

    # Write report
    print(f"Writing report to {args.output}")
    args.output.write_text(report)

    print("\nReport generated successfully!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Convert the completed s1 classifier metrics JSON into a concise Markdown report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _row(metrics: dict) -> str:
    return (
        f"| {metrics.get('samples', 0)} | {metrics.get('accuracy', 0.0):.4f} | "
        f"{metrics.get('precision_positive', 0.0):.4f} | "
        f"{metrics.get('recall_positive', 0.0):.4f} | "
        f"{metrics.get('f1_positive', 0.0):.4f} | {metrics.get('macro_f1', 0.0):.4f} |"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    metrics_path = args.metrics.resolve()
    output_path = (args.output or metrics_path.with_suffix(".md")).resolve()
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    best = payload.get("best_validation_metrics") or {}
    final = payload.get("final_validation_metrics") or {}
    lines = [
        "# S1 工程分类器训练评价指标",
        "",
        "## 训练配置",
        "",
        f"- 方法：`{payload.get('method', '')}`",
        f"- 设备：`{payload.get('device', '')}`",
        f"- 训练样本：`{payload.get('train_samples', 0)}`",
        f"- 验证样本：`{payload.get('validation_samples', 0)}`",
        f"- 最佳 epoch：`{payload.get('best_epoch', 0)}`",
        f"- 最佳模型：`{payload.get('classifier_checkpoint', '')}`",
        f"- SAM3 权重：`{payload.get('sam3_checkpoint', '')}`",
        f"- SAM3 状态：{payload.get('sam3_status', '')}",
        "",
        "## 验证集指标",
        "",
        "| 样本数 | Accuracy | Positive Precision | Positive Recall | Positive F1 | Macro-F1 |",
        "|---:|---:|---:|---:|---:|---:|",
        _row(best),
        "",
        "## 最后一轮指标",
        "",
        "| 样本数 | Accuracy | Positive Precision | Positive Recall | Positive F1 | Macro-F1 |",
        "|---:|---:|---:|---:|---:|---:|",
        _row(final),
        "",
        "## 混淆矩阵",
        "",
        "行是真实类别，列是预测类别；类别顺序为 `negative, positive`。",
        "",
        "```text",
        json.dumps(best.get("confusion_matrix", []), ensure_ascii=False),
        "```",
        "",
        "## 限制",
        "",
    ]
    for limitation in payload.get("limitations", []):
        lines.append(f"- {limitation}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Model Health Report Generator

This module generates a comprehensive model health report after training:
1. Collects all key metrics from training outputs
2. Creates a markdown report (.md)
3. Sends the report to Claude Code for interpretation
4. Generates a PDF with metrics + Claude's analysis

Usage:
    python model_report.py --run-dir <path_to_run_directory>
    python model_report.py  # Uses default generated/ directory
"""

import os
import sys
import json
import subprocess
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import re

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, ListFlowable, ListItem
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER


def sanitize_for_pdf(text: str) -> str:
    """Replace Unicode characters that Helvetica cannot render with ASCII equivalents."""
    replacements = {
        '\u2713': '[OK]',   # ✓
        '\u2717': '[X]',    # ✗
        '\u2714': '[OK]',   # ✔
        '\u2718': '[X]',    # ✘
        '\u2022': '-',      # •
        '\u2019': "'",      # '
        '\u2018': "'",      # '
        '\u201c': '"',      # "
        '\u201d': '"',      # "
        '\u2013': '-',      # –
        '\u2014': '--',     # —
        '\u2026': '...',    # …
        '\u2248': '~',      # ≈
        '\u2265': '>=',     # ≥
        '\u2264': '<=',     # ≤
        '\u2192': '->',     # →
        '\u2190': '<-',     # ←
        '\u20ac': 'EUR ',   # €
        '\u2705': '[OK]',   # ✅
        '\u26a0\ufe0f': '[!]',  # ⚠️
        '\u26a0': '[!]',    # ⚠
        '\u274c': '[X]',    # ❌
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text


def markdown_to_reportlab(text: str) -> str:
    """Convert markdown formatting to reportlab XML tags."""
    # Sanitize Unicode first
    text = sanitize_for_pdf(text)
    # Convert **bold** to <b>bold</b>
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    # Convert *italic* to <i>italic</i>
    text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
    # Convert `code` to <font name="Courier">code</font>
    text = re.sub(r'`(.+?)`', r'<font name="Courier">\1</font>', text)
    # Convert markdown bullet to dash
    text = text.replace('- ', '- ')
    # Escape XML special chars that aren't already tags
    text = text.replace('&', '&amp;')
    # Re-fix tags broken by & escaping
    text = text.replace('&amp;amp;', '&amp;')
    return text

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import ModelTrading.config.directories as dir_config


class ModelHealthReport:
    """Collects and reports model health metrics."""

    def __init__(self, run_dir: str):
        """
        Initialize report generator.

        Args:
            run_dir: Path to the run directory (e.g., generated/ or generated/{run_id}/)
        """
        self.run_dir = run_dir
        self.metrics = {}
        self.warnings = []
        self.status_checks = []

    def collect_metrics(self) -> Dict:
        """Collect all metrics from training outputs."""
        print("Collecting metrics from training outputs...")

        # 1. Training Summary
        self._collect_training_summary()

        # 2. Cross-Validation Results
        self._collect_cv_results()

        # 3. Feature Importance
        self._collect_feature_importance()

        # 4. Feature Selection Results
        self._collect_feature_selection()

        # 5. Backtest Results (if available)
        self._collect_backtest_results()

        # 6. Label Distribution
        self._collect_label_distribution()

        # Run health checks
        self._run_health_checks()

        return self.metrics

    def _collect_training_summary(self):
        """Load training_summary.json."""
        summary_path = os.path.join(self.run_dir, "training_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                self.metrics['training_summary'] = json.load(f)
            print(f"  Loaded: training_summary.json")
        else:
            self.warnings.append("training_summary.json not found")
            self.metrics['training_summary'] = {}

    def _collect_cv_results(self):
        """Extract CV results from training summary."""
        summary = self.metrics.get('training_summary', {})
        self.metrics['cv_results'] = {
            'mean_val_accuracy': summary.get('cv_mean_val_accuracy', None),
            'n_features_original': summary.get('n_features_original', None),
            'n_features_selected': summary.get('n_features_selected', None),
            'n_training_samples': summary.get('n_training_samples', None),
        }

    def _collect_feature_importance(self):
        """Load feature importance CSVs."""
        feature_map_dir = os.path.join(self.run_dir, "feature_map")
        if not os.path.exists(feature_map_dir):
            # Try alternate location
            feature_map_dir = os.path.join(os.path.dirname(self.run_dir), "training_output", "feature_map")

        self.metrics['feature_importance'] = {}

        if os.path.exists(feature_map_dir):
            for filename in os.listdir(feature_map_dir):
                if filename.endswith('.csv') and 'feature_importance' in filename:
                    model_name = filename.replace('feature_importance_', '').replace('.csv', '')
                    filepath = os.path.join(feature_map_dir, filename)
                    try:
                        df = pd.read_csv(filepath)
                        self.metrics['feature_importance'][model_name] = df.to_dict('records')
                        print(f"  Loaded: {filename}")
                    except Exception as e:
                        self.warnings.append(f"Failed to load {filename}: {e}")
        else:
            self.warnings.append("Feature importance directory not found")

    def _collect_feature_selection(self):
        """Load feature selection results (supports flat and fast/slow scoped layouts)."""
        fs_dir = os.path.join(self.run_dir, "feature_selection")
        self.metrics['feature_selection'] = {}

        if not os.path.exists(fs_dir):
            self.warnings.append("Feature selection directory not found")
            return

        # Determine scopes: use fast/slow subdirs if present, otherwise flat layout
        scopes = [s for s in ('fast', 'slow') if os.path.isdir(os.path.join(fs_dir, s))]
        if not scopes:
            scopes = [None]  # flat layout — no subdirectory

        for scope in scopes:
            scope_dir = os.path.join(fs_dir, scope) if scope else fs_dir
            key = scope if scope else 'default'
            self.metrics['feature_selection'][key] = {}
            label = f"[{scope}]" if scope else ""

            # MI scores
            mi_path = os.path.join(scope_dir, "mi_scores.csv")
            if os.path.exists(mi_path):
                df = pd.read_csv(mi_path)
                self.metrics['feature_selection'][key]['mi_scores'] = {
                    'top_5': df.head(5).to_dict('records'),
                    'bottom_5': df.tail(5).to_dict('records'),
                    'total_features': len(df)
                }
                print(f"  Loaded: {label} mi_scores.csv")

            # PFI scores
            pfi_path = os.path.join(scope_dir, "pfi_scores.csv")
            if os.path.exists(pfi_path):
                df = pd.read_csv(pfi_path)
                negative_features = df[df['importance'] < 0] if 'importance' in df.columns else pd.DataFrame()
                self.metrics['feature_selection'][key]['pfi_scores'] = {
                    'top_5': df.head(5).to_dict('records') if len(df) > 0 else [],
                    'negative_features': negative_features.to_dict('records') if len(negative_features) > 0 else [],
                    'n_negative': len(negative_features)
                }
                print(f"  Loaded: {label} pfi_scores.csv")

            # Selected features count
            selected_path = os.path.join(scope_dir, "selected_features.txt")
            if os.path.exists(selected_path):
                with open(selected_path, 'r') as f:
                    selected = [line.strip() for line in f if line.strip()]
                self.metrics['feature_selection'][key]['n_selected'] = len(selected)
                print(f"  Loaded: {label} selected_features.txt ({len(selected)} features)")

    def _collect_backtest_results(self):
        """Load backtest results if available."""
        # Try different possible locations
        report_dir = os.path.join(self.run_dir, "report")
        if not os.path.exists(report_dir):
            report_dir = os.path.join(os.path.dirname(self.run_dir), "report", "python")

        self.metrics['backtest'] = {}

        trade_list_path = os.path.join(report_dir, "trade_list.csv")
        if os.path.exists(trade_list_path):
            try:
                df = pd.read_csv(trade_list_path)

                # Calculate key metrics
                total_trades = len(df)
                if total_trades > 0:
                    winning_trades = len(df[df['pnl'] > 0])
                    win_rate = winning_trades / total_trades * 100
                    total_pnl = df['pnl'].sum()
                    total_pips = df['pnl_pips'].sum() if 'pnl_pips' in df.columns else 0
                    avg_pnl = df['pnl'].mean()
                    win_df = df[df['pnl'] > 0]
                    loss_df = df[df['pnl'] <= 0]
                    avg_win_pnl = float(win_df['pnl'].mean()) if len(win_df) > 0 else 0.0
                    avg_loss_pnl = float(loss_df['pnl'].mean()) if len(loss_df) > 0 else 0.0
                    max_pnl = df['pnl'].max()
                    min_pnl = df['pnl'].min()

                    # Exit reason breakdown
                    exit_reasons = df['exit_reason'].value_counts().to_dict() if 'exit_reason' in df.columns else {}

                    # Direction breakdown
                    if 'action' in df.columns:
                        direction_stats = {}
                        for action in df['action'].unique():
                            action_df = df[df['action'] == action]
                            direction_stats[action] = {
                                'count': len(action_df),
                                'win_rate': len(action_df[action_df['pnl'] > 0]) / len(action_df) * 100 if len(action_df) > 0 else 0,
                                'total_pnl': action_df['pnl'].sum()
                            }
                    else:
                        direction_stats = {}

                    self.metrics['backtest'] = {
                        'total_trades': total_trades,
                        'winning_trades': winning_trades,
                        'win_rate': win_rate,
                        'total_pnl': total_pnl,
                        'total_pips': total_pips,
                        'avg_pnl': avg_pnl,
                        'avg_win_pnl': avg_win_pnl,
                        'avg_loss_pnl': avg_loss_pnl,
                        'max_pnl': max_pnl,
                        'min_pnl': min_pnl,
                        'exit_reasons': exit_reasons,
                        'direction_stats': direction_stats
                    }
                    print(f"  Loaded: trade_list.csv ({total_trades} trades)")
            except Exception as e:
                self.warnings.append(f"Failed to load trade_list.csv: {e}")
        else:
            self.warnings.append("trade_list.csv not found - backtest may not have been run")

    def _collect_label_distribution(self):
        """Load label distribution from parquet files."""
        self.metrics['labels'] = {}

        # Try to load target parquet files
        for label_type in ['long_slow', 'short_slow', 'long_fast', 'short_fast']:
            # Different naming patterns
            patterns = [
                f"y_target_{label_type}.parquet",
                f"y_target_long_{label_type}.parquet" if 'long' in label_type else f"y_target_short_{label_type}.parquet",
            ]

            for pattern in patterns:
                filepath = os.path.join(self.run_dir, pattern)
                if os.path.exists(filepath):
                    try:
                        df = pd.read_parquet(filepath)
                        col = df.columns[0]
                        positive = (df[col] == 1).sum()
                        total = len(df)
                        self.metrics['labels'][label_type] = {
                            'positive': int(positive),
                            'total': int(total),
                            'ratio': positive / total * 100 if total > 0 else 0
                        }
                        print(f"  Loaded: {pattern}")
                        break
                    except Exception as e:
                        pass

    def _run_health_checks(self):
        """Run health checks and generate status indicators."""
        checks = []

        # 1. CV Accuracy Check
        cv_acc = self.metrics.get('cv_results', {}).get('mean_val_accuracy')
        if cv_acc is not None:
            if cv_acc > 0.55:
                checks.append(('CV Accuracy', 'PASS', f'{cv_acc:.1%} (>55%)'))
            elif cv_acc > 0.52:
                checks.append(('CV Accuracy', 'WARN', f'{cv_acc:.1%} (52-55%)'))
            else:
                checks.append(('CV Accuracy', 'FAIL', f'{cv_acc:.1%} (<52%)'))

        # 2. Win Rate Check
        win_rate = self.metrics.get('backtest', {}).get('win_rate')
        if win_rate is not None:
            if 50 <= win_rate <= 75:
                checks.append(('Win Rate', 'PASS', f'{win_rate:.1f}% (50-75%)'))
            elif win_rate > 80:
                checks.append(('Win Rate', 'WARN', f'{win_rate:.1f}% (>80% - suspicious)'))
            else:
                checks.append(('Win Rate', 'FAIL', f'{win_rate:.1f}% (<50%)'))

        # 3. Total PnL Check
        total_pnl = self.metrics.get('backtest', {}).get('total_pnl')
        if total_pnl is not None:
            if total_pnl > 0:
                checks.append(('Total PnL', 'PASS', f'€{total_pnl:,.2f}'))
            else:
                checks.append(('Total PnL', 'FAIL', f'€{total_pnl:,.2f}'))

        # 4. Trade Count Check
        total_trades = self.metrics.get('backtest', {}).get('total_trades')
        if total_trades is not None:
            if total_trades >= 30:
                checks.append(('Trade Count', 'PASS', f'{total_trades} trades (≥30)'))
            elif total_trades >= 15:
                checks.append(('Trade Count', 'WARN', f'{total_trades} trades (15-30)'))
            else:
                checks.append(('Trade Count', 'FAIL', f'{total_trades} trades (<15)'))

        # 5. Stop Loss Rate Check
        exit_reasons = self.metrics.get('backtest', {}).get('exit_reasons', {})
        if exit_reasons and total_trades:
            stop_loss_count = exit_reasons.get('stop_loss', 0)
            stop_loss_rate = stop_loss_count / total_trades * 100
            if stop_loss_rate < 30:
                checks.append(('Stop Loss Rate', 'PASS', f'{stop_loss_rate:.1f}% (<30%)'))
            elif stop_loss_rate < 50:
                checks.append(('Stop Loss Rate', 'WARN', f'{stop_loss_rate:.1f}% (30-50%)'))
            else:
                checks.append(('Stop Loss Rate', 'FAIL', f'{stop_loss_rate:.1f}% (>50%)'))

        # 6. Direction Balance Check
        direction_stats = self.metrics.get('backtest', {}).get('direction_stats', {})
        if direction_stats:
            buy_pnl = direction_stats.get('BUY', {}).get('total_pnl', 0)
            sell_pnl = direction_stats.get('SELL', {}).get('total_pnl', 0)
            if buy_pnl > 0 and sell_pnl > 0:
                checks.append(('Direction Balance', 'PASS', f'BUY: €{buy_pnl:,.0f}, SELL: €{sell_pnl:,.0f}'))
            elif buy_pnl > 0 or sell_pnl > 0:
                checks.append(('Direction Balance', 'WARN', f'BUY: €{buy_pnl:,.0f}, SELL: €{sell_pnl:,.0f}'))
            else:
                checks.append(('Direction Balance', 'FAIL', f'Both directions negative'))

        # 7. Feature Selection Check
        n_original = self.metrics.get('cv_results', {}).get('n_features_original')
        n_selected = self.metrics.get('cv_results', {}).get('n_features_selected')
        if n_original and n_selected:
            reduction = (1 - n_selected / n_original) * 100
            if reduction < 50:
                checks.append(('Feature Selection', 'PASS', f'{n_selected}/{n_original} ({reduction:.0f}% reduced)'))
            else:
                checks.append(('Feature Selection', 'WARN', f'{n_selected}/{n_original} ({reduction:.0f}% reduced - aggressive)'))

        self.status_checks = checks

    def generate_markdown(self) -> str:
        """Generate markdown report."""
        lines = []

        # Header
        lines.append("# Model Health Report")
        lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Run Directory:** `{self.run_dir}`\n")

        # Health Check Summary
        lines.append("## Health Check Summary\n")
        lines.append("| Check | Status | Value |")
        lines.append("|-------|--------|-------|")
        for check_name, status, value in self.status_checks:
            status_icon = "✅" if status == "PASS" else "⚠️" if status == "WARN" else "❌"
            lines.append(f"| {check_name} | {status_icon} {status} | {value} |")
        lines.append("")

        # Training Summary
        summary = self.metrics.get('training_summary', {})
        if summary:
            lines.append("## Training Configuration\n")
            lines.append(f"- **Training Period:** {summary.get('train_start', 'N/A')} to {summary.get('train_end', 'N/A')}")
            lines.append(f"- **Training Samples:** {summary.get('n_training_samples', 'N/A'):,}")
            lines.append(f"- **Original Features:** {summary.get('n_features_original', 'N/A')}")
            lines.append(f"- **Selected Features:** {summary.get('n_features_selected', 'N/A')}")
            lines.append(f"- **CV Validation Accuracy:** {summary.get('cv_mean_val_accuracy', 0):.2%}")
            lines.append(f"- **MI Threshold:** {summary.get('mi_threshold', 'N/A')}")
            lines.append(f"- **PFI Threshold:** {summary.get('pfi_threshold', 'N/A')}")
            lines.append("")

        # Backtest Results
        backtest = self.metrics.get('backtest', {})
        if backtest:
            lines.append("## Backtest Results\n")
            lines.append("### Performance Metrics\n")
            lines.append(f"- **Total Trades:** {backtest.get('total_trades', 0)}")
            lines.append(f"- **Win Rate:** {backtest.get('win_rate', 0):.1f}%")
            lines.append(f"- **Total PnL:** €{backtest.get('total_pnl', 0):,.2f}")
            lines.append(f"- **Total Pips:** {backtest.get('total_pips', 0):,.1f}")
            lines.append(f"- **Avg PnL/Trade:** €{backtest.get('avg_pnl', 0):,.2f}")
            lines.append(f"- **Avg Winning PnL/Trade:** €{backtest.get('avg_win_pnl', 0):,.2f}")
            lines.append(f"- **Avg Losing PnL/Trade:** €{backtest.get('avg_loss_pnl', 0):,.2f}")
            lines.append(f"- **Best Trade:** €{backtest.get('max_pnl', 0):,.2f}")
            lines.append(f"- **Worst Trade:** €{backtest.get('min_pnl', 0):,.2f}")
            lines.append("")

            # Exit Reasons
            exit_reasons = backtest.get('exit_reasons', {})
            if exit_reasons:
                lines.append("### Exit Reasons\n")
                lines.append("| Reason | Count | Percentage |")
                lines.append("|--------|-------|------------|")
                total = sum(exit_reasons.values())
                for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
                    pct = count / total * 100 if total > 0 else 0
                    lines.append(f"| {reason} | {count} | {pct:.1f}% |")
                lines.append("")

            # Direction Stats
            direction_stats = backtest.get('direction_stats', {})
            if direction_stats:
                lines.append("### Performance by Direction\n")
                lines.append("| Direction | Trades | Win Rate | Total PnL |")
                lines.append("|-----------|--------|----------|-----------|")
                for direction, stats in direction_stats.items():
                    lines.append(f"| {direction} | {stats['count']} | {stats['win_rate']:.1f}% | €{stats['total_pnl']:,.2f} |")
                lines.append("")

        # Feature Importance
        feature_imp = self.metrics.get('feature_importance', {})
        if feature_imp:
            lines.append("## Features by Model\n")
            for model_name, features in feature_imp.items():
                lines.append(f"### {model_name}\n")
                lines.append("| Rank | Feature | Importance |")
                lines.append("|------|---------|------------|")
                for i, feat in enumerate(features, 1):
                    feat_name = feat.get('Feature', feat.get('feature', 'N/A'))
                    importance = feat.get('Gain', feat.get('gain', feat.get('importance', 0)))
                    lines.append(f"| {i} | {feat_name} | {importance:.4f} |")
                lines.append("")

        # Warnings
        if self.warnings:
            lines.append("## Warnings\n")
            for warning in self.warnings:
                lines.append(f"- ⚠️ {warning}")
            lines.append("")

        # Request for Claude Analysis
        lines.append("---\n")
        lines.append("## Analysis Request\n")
        lines.append("Please analyze this model health report and provide:\n")
        lines.append("1. **Overall Assessment:** Is this model ready for live trading?")
        lines.append("2. **Key Concerns:** What are the main issues or risks?")
        lines.append("3. **Recommendations:** What improvements would you suggest?")
        lines.append("4. **Feature Analysis:** Do the top features make sense for forex trading?")
        lines.append("5. **Exit Strategy:** Is the exit reason distribution healthy?")

        return "\n".join(lines)

    def save_markdown(self, output_path: Optional[str] = None) -> str:
        """Save markdown report to file."""
        if output_path is None:
            output_path = os.path.join(self.run_dir, "model_health_report.md")

        md_content = self.generate_markdown()
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)

        print(f"Markdown report saved to: {output_path}")
        return output_path

    def get_claude_analysis(self, md_path: str) -> str:
        """Send markdown to Claude Code and get interpretation."""
        print("\nSending report to Claude Code for analysis...")

        # Read the markdown content
        with open(md_path, 'r', encoding='utf-8') as f:
            md_content = f.read()

        # Prepare the prompt
        prompt = f"""Please analyze this forex trading model health report and provide your expert interpretation.

{md_content}

Provide a structured analysis with:
1. Overall Assessment (1-2 sentences)
2. Key Concerns (bullet points)
3. Specific Recommendations (bullet points)
4. Feature Analysis (are the top features sensible?)
5. Risk Assessment (what could go wrong in live trading?)

Be concise but thorough. Focus on actionable insights."""

        try:
            # Call Claude Code CLI
            result = subprocess.run(
                ['claude', '-p', prompt],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=self.run_dir
            )

            if result.returncode == 0:
                analysis = result.stdout.strip()
                print("Claude analysis received successfully.")
                return analysis
            else:
                error_msg = f"Claude Code returned error: {result.stderr}"
                print(error_msg)
                return f"Error getting Claude analysis: {error_msg}"

        except subprocess.TimeoutExpired:
            return "Error: Claude Code analysis timed out after 120 seconds."
        except FileNotFoundError:
            return "Error: Claude Code CLI not found. Please ensure 'claude' is installed and in PATH."
        except Exception as e:
            return f"Error calling Claude Code: {str(e)}"

    def generate_pdf(self, claude_analysis: str, output_path: Optional[str] = None) -> str:
        """Generate PDF report with metrics and Claude analysis."""
        if output_path is None:
            output_path = os.path.join(self.run_dir, "model_health_report.pdf")

        print(f"\nGenerating PDF report...")

        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=1.5*cm,
            leftMargin=1.5*cm,
            topMargin=1.5*cm,
            bottomMargin=1.5*cm
        )

        styles = getSampleStyleSheet()

        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=12,
            textColor=colors.darkblue
        )

        h2_style = ParagraphStyle(
            'CustomH2',
            parent=styles['Heading2'],
            fontSize=14,
            spaceBefore=12,
            spaceAfter=6,
            textColor=colors.darkblue
        )

        h3_style = ParagraphStyle(
            'CustomH3',
            parent=styles['Heading3'],
            fontSize=12,
            spaceBefore=8,
            spaceAfter=4
        )

        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=6
        )

        elements = []

        # Title
        elements.append(Paragraph("Model Health Report", title_style))
        elements.append(Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            body_style
        ))
        elements.append(Spacer(1, 0.3*inch))

        # Health Check Summary Table
        elements.append(Paragraph("Health Check Summary", h2_style))

        check_data = [['Check', 'Status', 'Value']]
        for check_name, status, value in self.status_checks:
            status_text = f"{'[OK]' if status == 'PASS' else '[!]' if status == 'WARN' else '[X]'} {status}"
            check_data.append([check_name, status_text, sanitize_for_pdf(value)])

        check_table = Table(check_data, colWidths=[2.5*inch, 1*inch, 3*inch])
        check_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('TOPPADDING', (0, 1), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 4),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.Color(0.95, 0.95, 0.95)])
        ]))
        elements.append(check_table)
        elements.append(Spacer(1, 0.3*inch))

        # Training Configuration
        summary = self.metrics.get('training_summary', {})
        if summary:
            elements.append(Paragraph("Training Configuration", h2_style))
            config_text = f"""
            <b>Training Period:</b> {summary.get('train_start', 'N/A')} to {summary.get('train_end', 'N/A')}<br/>
            <b>Training Samples:</b> {summary.get('n_training_samples', 'N/A'):,}<br/>
            <b>Features:</b> {summary.get('n_features_selected', 'N/A')} selected from {summary.get('n_features_original', 'N/A')}<br/>
            <b>CV Validation Accuracy:</b> {summary.get('cv_mean_val_accuracy', 0):.2%}
            """
            elements.append(Paragraph(config_text, body_style))
            elements.append(Spacer(1, 0.2*inch))

        # Backtest Results
        backtest = self.metrics.get('backtest', {})
        if backtest:
            elements.append(Paragraph("Backtest Performance", h2_style))

            perf_data = [
                ['Metric', 'Value'],
                ['Total Trades', str(backtest.get('total_trades', 0))],
                ['Win Rate', f"{backtest.get('win_rate', 0):.1f}%"],
                ['Total PnL', f"EUR {backtest.get('total_pnl', 0):,.2f}"],
                ['Total Pips', f"{backtest.get('total_pips', 0):,.1f}"],
                ['Avg PnL/Trade', f"EUR {backtest.get('avg_pnl', 0):,.2f}"],
                ['Avg Winning PnL/Trade', f"EUR {backtest.get('avg_win_pnl', 0):,.2f}"],
                ['Avg Losing PnL/Trade', f"EUR {backtest.get('avg_loss_pnl', 0):,.2f}"],
                ['Best Trade', f"EUR {backtest.get('max_pnl', 0):,.2f}"],
                ['Worst Trade', f"EUR {backtest.get('min_pnl', 0):,.2f}"],
            ]

            perf_table = Table(perf_data, colWidths=[2*inch, 2*inch])
            perf_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('ALIGN', (1, 1), (1, -1), 'RIGHT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.Color(0.95, 0.95, 0.95)])
            ]))
            elements.append(perf_table)
            elements.append(Spacer(1, 0.2*inch))

            # Exit Reasons
            exit_reasons = backtest.get('exit_reasons', {})
            if exit_reasons:
                elements.append(Paragraph("Exit Reasons", h3_style))
                total = sum(exit_reasons.values())
                exit_data = [['Reason', 'Count', '%']]
                for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
                    pct = count / total * 100 if total > 0 else 0
                    exit_data.append([reason, str(count), f"{pct:.1f}%"])

                exit_table = Table(exit_data, colWidths=[2.5*inch, 1*inch, 1*inch])
                exit_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('FONTSIZE', (0, 0), (-1, -1), 8),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ]))
                elements.append(exit_table)

        elements.append(PageBreak())

        # Claude Analysis Section
        elements.append(Paragraph("AI Analysis (Claude)", h2_style))
        elements.append(Spacer(1, 0.1*inch))

        # Split Claude's analysis into paragraphs
        analysis_paragraphs = claude_analysis.split('\n\n')
        for para in analysis_paragraphs:
            if para.strip():
                # Handle markdown-style headers
                stripped = para.strip()
                if stripped.startswith('#'):
                    header_text = stripped.lstrip('#').strip()
                    header_text = sanitize_for_pdf(header_text)
                    elements.append(Paragraph(header_text, h3_style))
                else:
                    converted = markdown_to_reportlab(stripped)
                    converted = converted.replace('\n', '<br/>')
                    try:
                        elements.append(Paragraph(converted, body_style))
                    except Exception:
                        # Fallback: strip all XML tags if parsing fails
                        plain = re.sub(r'<[^>]+>', '', converted)
                        elements.append(Paragraph(plain, body_style))
                elements.append(Spacer(1, 0.1*inch))

        # Build PDF
        doc.build(elements)
        print(f"PDF report saved to: {output_path}")
        return output_path


def generate_report(run_dir: str = None, skip_claude: bool = False) -> Tuple[str, str, str]:
    """
    Generate complete model health report.

    Args:
        run_dir: Path to run directory. If None, uses default generated/.
        skip_claude: If True, skip Claude analysis (for testing).

    Returns:
        Tuple of (md_path, pdf_path, claude_analysis)
    """
    if run_dir is None:
        run_dir = dir_config.GENERATED_DIR

    print("=" * 80)
    print("MODEL HEALTH REPORT GENERATOR")
    print("=" * 80)
    print(f"Run directory: {run_dir}\n")

    report = ModelHealthReport(run_dir)

    # Collect metrics
    report.collect_metrics()

    # Generate and save markdown
    md_path = report.save_markdown()

    # Get Claude analysis
    if skip_claude:
        claude_analysis = "Claude analysis skipped (--skip-claude flag)."
    else:
        claude_analysis = report.get_claude_analysis(md_path)

    # Save Claude analysis to separate file
    analysis_path = os.path.join(run_dir, "claude_analysis.md")
    with open(analysis_path, 'w', encoding='utf-8') as f:
        f.write("# Claude Analysis\n\n")
        f.write(claude_analysis)
    print(f"Claude analysis saved to: {analysis_path}")

    # Generate PDF
    pdf_path = report.generate_pdf(claude_analysis)

    print("\n" + "=" * 80)
    print("REPORT GENERATION COMPLETE")
    print("=" * 80)
    print(f"Markdown: {md_path}")
    print(f"Analysis: {analysis_path}")
    print(f"PDF: {pdf_path}")

    return md_path, pdf_path, claude_analysis


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Generate model health report')
    parser.add_argument('--run-dir', type=str, default=None,
                        help='Path to run directory (default: generated/)')
    parser.add_argument('--skip-claude', action='store_true',
                        help='Skip Claude analysis (for testing)')

    args = parser.parse_args()
    generate_report(args.run_dir, args.skip_claude)


if __name__ == '__main__':
    main()

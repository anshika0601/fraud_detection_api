#!/usr/bin/env python
"""
Drift detection script: runs on recent predictions, generates HTML report.
Usage: python run_drift_detection.py [--n-rows 1000] [--output reports/drift_report.html]
"""

import argparse
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from src.monitoring.drift_detector import DriftDetector, load_reference_data

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_html_summary(summary: dict, drift_metrics: dict, output_path: str) -> None:
    """Generate a summary HTML report."""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fraud Detection - Drift Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            h1 {{ color: #333; border-bottom: 3px solid #0066cc; padding-bottom: 10px; }}
            h2 {{ color: #0066cc; margin-top: 30px; }}
            .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }}
            .metric-card {{ background: #f9f9f9; padding: 15px; border-left: 4px solid #0066cc; border-radius: 4px; }}
            .metric-label {{ font-weight: bold; color: #666; font-size: 0.9em; }}
            .metric-value {{ font-size: 1.8em; color: #0066cc; margin: 10px 0; }}
            .alert {{ padding: 15px; margin: 15px 0; border-radius: 4px; }}
            .alert-danger {{ background: #f8d7da; color: #721c24; border-left: 4px solid #f5c6cb; }}
            .alert-success {{ background: #d4edda; color: #155724; border-left: 4px solid #c3e6cb; }}
            .alert-info {{ background: #d1ecf1; color: #0c5460; border-left: 4px solid #bee5eb; }}
            .timestamp {{ color: #999; font-size: 0.85em; margin-top: 20px; padding-top: 10px; border-top: 1px solid #eee; }}
            .report-link {{ display: inline-block; margin-top: 20px; padding: 10px 20px; background: #0066cc; color: white; text-decoration: none; border-radius: 4px; }}
            .report-link:hover {{ background: #0052a3; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 Fraud Detection - Data Drift Report</h1>

            <div class="metrics">
                <div class="metric-card">
                    <div class="metric-label">Reference Data Rows</div>
                    <div class="metric-value">{summary['reference_rows']:,}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Current Data Rows</div>
                    <div class="metric-value">{summary['current_rows']:,}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Report Generated</div>
                    <div class="metric-value">{datetime.fromisoformat(summary['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}</div>
                </div>
            </div>

            <h2>Prediction Distribution</h2>
            <div class="metrics">
                <div class="metric-card">
                    <div class="metric-label">Reference Fraud Rate</div>
                    <div class="metric-value">{drift_metrics['reference_fraud_rate']*100:.2f}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Current Fraud Rate</div>
                    <div class="metric-value">{drift_metrics['current_fraud_rate']*100:.2f}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Drift Magnitude</div>
                    <div class="metric-value">{drift_metrics['drift_percentage']*100:.2f}%</div>
                </div>
            </div>

            <h2>Drift Status</h2>
            {'<div class="alert alert-danger">🚨 <strong>DRIFT DETECTED</strong> - Fraud rate has shifted beyond threshold (' + str(drift_metrics['threshold']*100) + '%)</div>' if drift_metrics['is_drifted'] else '<div class="alert alert-success">✓ <strong>No Drift Detected</strong> - Distribution remains stable</div>'}

            <h2>Detailed Analysis</h2>
            <p>For detailed feature-level drift analysis, see the comprehensive Evidently report:</p>
            <a href="{Path(summary['report_path']).name}" class="report-link">View Full Report →</a>

            <div class="timestamp">
                Report generated on {datetime.fromisoformat(summary['timestamp']).strftime('%Y-%m-%d %H:%M:%S UTC')}
            </div>
        </div>
    </body>
    </html>
    """

    with open(output_path, 'w') as f:
        f.write(html)
    logger.info(f"Summary report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run drift detection on recent predictions"
    )
    parser.add_argument(
        "--n-rows",
        type=int,
        default=1000,
        help="Number of recent predictions to analyze (default: 1000)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/drift_report.html",
        help="Output path for HTML report (default: reports/drift_report.html)"
    )
    parser.add_argument(
        "--reference-data",
        type=str,
        default="data/train_processed.csv",
        help="Path to reference training data"
    )
    parser.add_argument(
        "--predictions-csv",
        type=str,
        default="logs/predictions.csv",
        help="Path to predictions CSV"
    )
    args = parser.parse_args()

    try:
        # Load reference data
        logger.info("Loading reference data...")
        ref_data = load_reference_data(args.reference_data)

        # Initialize detector
        detector = DriftDetector(ref_data, args.predictions_csv)

        # Load recent predictions
        logger.info(f"Loading recent {args.n_rows} predictions...")
        current_data = detector.load_recent_predictions(n_rows=args.n_rows)

        # Generate drift report
        logger.info("Generating drift detection report...")
        report, summary = detector.generate_drift_report(
            current_data,
            output_path=args.output
        )

        # Detect prediction drift
        predictions = current_data['prediction'].astype(int)
        drift_metrics = detector.detect_prediction_drift(predictions)

        # Generate summary HTML
        summary_path = args.output.replace('.html', '_summary.html')
        generate_html_summary(summary, drift_metrics, summary_path)

        # Log results
        logger.info(f"✓ Drift detection complete")
        logger.info(f"  Reference fraud rate: {drift_metrics['reference_fraud_rate']*100:.2f}%")
        logger.info(f"  Current fraud rate: {drift_metrics['current_fraud_rate']*100:.2f}%")
        logger.info(f"  Drift: {drift_metrics['drift_percentage']*100:.2f}%")
        logger.info(f"  Status: {'DRIFTED ⚠️' if drift_metrics['is_drifted'] else 'STABLE ✓'}")
        logger.info(f"  Full report: {args.output}")
        logger.info(f"  Summary report: {summary_path}")

    except Exception as e:
        logger.error(f"Error running drift detection: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

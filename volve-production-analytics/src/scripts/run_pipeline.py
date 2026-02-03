"""
Run Pipeline Script
===================

Execute the full production analytics pipeline:
1. Load and prepare raw data
2. Engineer features
3. Generate forecasts
4. Run backtesting evaluation
5. Generate email summary report
6. Sync outputs to SharePoint (optional)

Usage:
    python -m src.scripts.run_pipeline --input data/raw/volve_production.csv
    python -m src.scripts.run_pipeline  # uses default location
    python -m src.scripts.run_pipeline --sync-sharepoint  # sync to SharePoint
"""

import argparse
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, SHAREPOINT_SITE_URL
from src.data_prep import prepare_data, load_processed_data
from src.features import engineer_features
from src.forecasting import forecast_series, forecast_all_wellbores
from src.evaluation import evaluate_models, save_metrics
from src.reporting import save_email_summary
from src.io_sharepoint import SharePointClient, sync_to_sharepoint


def find_raw_data_file() -> Path:
    """Find the raw Volve production data file."""
    # Check common locations
    possible_paths = [
        RAW_DATA_DIR / "Volve production data.csv",
        RAW_DATA_DIR / "volve_production.csv",
        RAW_DATA_DIR / "volve_data.csv",
        # Also check parent course directory
        project_root.parent / "Course Notebooks" / "Data" / "Volve production data.csv",
    ]

    for path in possible_paths:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Could not find raw data file. Please place Volve production data in: "
        f"{RAW_DATA_DIR}/Volve production data.csv"
    )


def run_pipeline(input_path: Path = None, verbose: bool = True, sync_sharepoint: bool = False) -> dict:
    """
    Run the complete analytics pipeline.

    Parameters
    ----------
    input_path : Path, optional
        Path to raw data file.
    verbose : bool
        Whether to print progress messages.

    Returns
    -------
    dict
        Pipeline results summary.
    """
    results = {}
    total_steps = 6 if sync_sharepoint else 5

    # Step 1: Find and prepare data
    if verbose:
        print("=" * 60)
        print("VOLVE PRODUCTION ANALYTICS PIPELINE")
        print("=" * 60)
        print(f"\n[1/{total_steps}] Preparing data...")

    if input_path is None:
        input_path = find_raw_data_file()

    if verbose:
        print(f"      Input: {input_path}")

    df = prepare_data(input_path, save_output=True)
    results["n_records"] = len(df)
    results["n_wellbores"] = df["wellbore"].nunique()
    results["date_range"] = f"{df['date'].min()} to {df['date'].max()}"

    if verbose:
        print(f"      Records: {results['n_records']}")
        print(f"      Wellbores: {results['n_wellbores']}")
        print(f"      Date range: {results['date_range']}")

    # Step 2: Engineer features
    if verbose:
        print(f"\n[2/{total_steps}] Engineering features...")

    df_features = engineer_features(df)

    if verbose:
        print(f"      Features added: rolling averages, YoY/MoM changes, uptime rate")

    # Step 3: Generate forecasts
    if verbose:
        print(f"\n[3/{total_steps}] Generating forecasts...")

    # Forecast total production with both models
    baseline_forecast = forecast_series(df, target_col="oil", series_id="TOTAL", model="baseline")
    ets_forecast = forecast_series(df, target_col="oil", series_id="TOTAL", model="ets")

    # Save forecasts
    all_forecasts = forecast_all_wellbores(df, target_col="oil", model="ets")
    all_forecasts.to_csv(PROCESSED_DATA_DIR / "forecasts.csv", index=False)

    results["forecast_horizon"] = len(ets_forecast)

    if verbose:
        print(f"      Forecast horizon: {results['forecast_horizon']} months")
        print(f"      Next month ETS forecast: {ets_forecast['yhat'].iloc[0]:,.0f} Sm3 oil")

    # Step 4: Backtest evaluation
    if verbose:
        print(f"\n[4/{total_steps}] Running backtest evaluation...")

    metrics_df = evaluate_models(df, target_col="oil", series_ids=["TOTAL"])
    save_metrics(metrics_df)

    # Get total field metrics
    total_metrics = metrics_df[metrics_df["series_id"] == "TOTAL"]
    results["metrics"] = total_metrics.to_dict(orient="records")

    if verbose:
        print("      Model Comparison (Total Field):")
        for _, row in total_metrics.iterrows():
            print(f"        {row['model']}: MAE={row['mae']:,.0f}, MAPE={row['mape']:.1f}%")

    # Step 5: Generate email summary
    if verbose:
        print(f"\n[5/{total_steps}] Generating email summary...")

    summary_text = save_email_summary(df_features, ets_forecast)

    if verbose:
        print(f"      Summary saved to: {PROCESSED_DATA_DIR / 'email_summary.txt'}")

    # Step 6: Sync to SharePoint (optional)
    if sync_sharepoint:
        if verbose:
            print(f"\n[6/{total_steps}] Syncing outputs to SharePoint...")

        use_sharepoint = bool(SHAREPOINT_SITE_URL)
        try:
            client = SharePointClient(use_sharepoint=use_sharepoint)
            uploaded = sync_to_sharepoint(
                client=client,
                local_folder=PROCESSED_DATA_DIR,
                remote_folder="Pipeline Outputs",
                file_pattern="*.csv",
            )
            results["sharepoint_synced"] = uploaded
            mode_label = "SharePoint" if use_sharepoint else "local fallback"
            if verbose:
                print(f"      Mode: {mode_label}")
                print(f"      Synced {len(uploaded)} file(s): {uploaded}")
        except (ValueError, RuntimeError, ImportError) as e:
            # ValueError: missing credentials
            # RuntimeError: token acquisition failed
            # ImportError: msal/requests not installed
            results["sharepoint_synced"] = []
            if verbose:
                print(f"      SharePoint sync failed: {e}")
                print("      Pipeline results are still available locally.")
        except OSError as e:
            # Network or filesystem errors during sync
            results["sharepoint_synced"] = []
            if verbose:
                print(f"      SharePoint sync failed (I/O): {e}")
                print("      Pipeline results are still available locally.")

    # Final output
    if verbose:
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        print(f"\nOutputs saved to: {PROCESSED_DATA_DIR}")
        print("  - volve_monthly.parquet")
        print("  - volve_monthly.csv")
        print("  - forecasts.csv")
        print("  - metrics.json")
        print("  - email_summary.txt")
        if sync_sharepoint:
            print("  - (synced to SharePoint or local fallback)")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run Volve Production Analytics Pipeline"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Path to raw production data CSV"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress messages"
    )
    parser.add_argument(
        "--sync-sharepoint",
        action="store_true",
        default=False,
        help="Sync pipeline outputs to SharePoint (requires Azure AD credentials; falls back to local copy)"
    )

    args = parser.parse_args()

    input_path = Path(args.input) if args.input else None
    run_pipeline(input_path=input_path, verbose=not args.quiet, sync_sharepoint=args.sync_sharepoint)


if __name__ == "__main__":
    main()

import os
import shutil
import logging
from pathlib import Path
import kagglehub

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def download_dataset(output_dir: str = "data/raw") -> None:
    """Download the Customer Support on Twitter dataset."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = Path(output_dir) / "twcs.csv"
    
    if output_path.exists():
        logger.info(f"Dataset already exists at {output_path}")
        return

    logger.info("Attempting to download 'thoughtvector/customer-support-on-twitter' via kagglehub...")
    try:
        # download returns the path to the directory containing the dataset files
        dataset_path = kagglehub.dataset_download("thoughtvector/customer-support-on-twitter")
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        logger.error("Please ensure you have Kaggle credentials configured (e.g. ~/.kaggle/kaggle.json or env vars).")
        raise
        
    logger.info(f"Dataset downloaded to {dataset_path}")
    
    # Locate the CSV
    csv_files = list(Path(dataset_path).rglob("*.csv"))
    if not csv_files:
        logger.error(f"No CSV files found in downloaded path: {dataset_path}")
        raise FileNotFoundError("Could not find twcs.csv")
    
    source_csv = None
    # Prefer twcs.csv if present
    for f in csv_files:
        if f.name == "twcs.csv":
            source_csv = f
            break
            
    # Fallback to any CSV if twcs.csv is not explicitly found
    if not source_csv:
        source_csv = csv_files[0]
        
    logger.info(f"Copying {source_csv} to {output_path}")
    shutil.copy(source_csv, output_path)
    logger.info("Download and extraction complete.")

if __name__ == "__main__":
    download_dataset()

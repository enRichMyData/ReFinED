import pandas as pd
import sys
from my_tests.utility.test_utils import get_latest_log, green_info, bolden


# Helpers
# ============================================================

def load_log(model_name="wikipedia_model_with_numbers") -> pd.DataFrame:
    """Loads and cleans the experiment log for a given model."""
    log_file = get_latest_log(model_name)
    print(green_info(f"[INFO] Loading log: {log_file}"))
    try:
        df = pd.read_csv(log_file, skip_blank_lines=True, on_bad_lines='skip')
    except FileNotFoundError:
        print(f"[ERROR] Log file not found: {log_file}")
        return pd.DataFrame()

    # Keep only rows that look like timestamps
    df = df[df['timestamp'].str.contains(r'\d{4}-\d{2}-\d{2}', na=False, regex=True)]

    # Convert numeric columns
    numeric_cols = ['accuracy', 'precision', 'recall', 'f1_score',
                    'throughput_s', 'peak_vram_gb', 'total_time_s',
                    'tp', 'fp', 'fn', 'n_samples_tested']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['accuracy'])
    return df


def separator(width: int, char: str = '-') -> str:
    return char * width


def print_table(df_table: pd.DataFrame, group_col: str = None):
    """Prints a DataFrame as a formatted table with optional group separators."""
    lines = df_table.to_string(index=False).split('\n')
    header, rows = lines[0], lines[1:]
    width = len(header)
    print(header)
    print(separator(width))
    if group_col and group_col in df_table.columns:
        col_values = df_table[group_col].tolist()
        for i, (row, val) in enumerate(zip(rows, col_values)):
            print(row)
            if i < len(rows) - 1 and val != col_values[i + 1]:
                print(separator(width))
    else:
        for row in rows:
            print(row)
    print(separator(width))


# Analysis Sections
# ============================================================

# prints highlights
def print_highlights(df: pd.DataFrame):
    best_acc  = df.loc[df['accuracy'].idxmax()]
    fastest   = df.loc[df['throughput_s'].idxmax()]
    struggler = df.loc[df['f1_score'].idxmin()]
    best_f1   = df.loc[df['f1_score'].idxmax()]
    print(bolden(f"{'Top Accuracy:':<16}") + f"{best_acc['accuracy']:.2%} | {best_acc['dataset']} ({best_acc['mode']})")
    print(bolden(f"{'Best F1:':<16}") + f"{best_f1['f1_score']:.4f} | {best_f1['dataset']} ({best_f1['mode']})")
    print(bolden(f"{'Peak Speed:':<16}") + f"{fastest['throughput_s']:.1f} tx/s | {fastest['dataset']} ({fastest['mode']})")
    print(bolden(f"{'Hardest Task:':<16}") + f"{struggler['dataset']} ({struggler['mode']}) — F1: {struggler['f1_score']:.4f}")


# prints summary of total run
def print_summary(df: pd.DataFrame):
    print(bolden(f"\nRun Summary:"))
    print(f"   - Datasets Logged:  {len(df)}")
    print(f"   - Average F1:       {df['f1_score'].mean():.4f}")
    print(f"   - Average Precision:{df['precision'].mean():.4f}")
    print(f"   - Average Recall:   {df['recall'].mean():.4f}")
    if 'peak_vram_gb' in df.columns:
        print(f"   - Max VRAM Used:    {df['peak_vram_gb'].max():.2f} GB")
    if 'total_time_s' in df.columns:
        total_hours = df['total_time_s'].sum() / 3600
        print(f"   - Total Inference:  {total_hours:.2f} hours")


# prints full results
def print_full_results(df: pd.DataFrame):
    print(bolden(f"\nFull Results:"))
    table = df[['dataset', 'mode', 'accuracy', 'precision', 'recall', 'f1_score', 'throughput_s']].copy()
    table = table.sort_values(['mode', 'dataset'])
    print_table(table, group_col='mode')


# prints best F1 per dataset across modes
def print_best_per_dataset(df: pd.DataFrame):
    print(bolden(f"\nBest F1 per dataset:"))
    best = df.loc[df.groupby('dataset')['f1_score'].idxmax()]
    best = best[['dataset', 'mode', 'accuracy', 'precision', 'recall', 'f1_score']]
    best = best.sort_values('f1_score', ascending=False).reset_index(drop=True)
    print_table(best)


# prints cell vs row comparison for datasets with both
def print_cell_vs_row(df: pd.DataFrame):
    cell = df[df['mode'] == 'cell'][['dataset', 'f1_score']].rename(columns={'f1_score': 'cell_f1'})
    row  = df[df['mode'] == 'row'][['dataset', 'f1_score']].rename(columns={'f1_score': 'row_f1'})
    merged = cell.merge(row, on='dataset', how='inner')
    if merged.empty: return
    merged['delta'] = (merged['row_f1'] - merged['cell_f1']).round(4)
    merged['winner'] = merged['delta'].apply(lambda x: 'row ▲' if x > 0 else 'cell ▼')
    merged = merged.sort_values('delta', ascending=False).reset_index(drop=True)
    print(bolden(f"\nCell vs Row Comparison (datasets with both modes):"))
    print_table(merged)


# prints VRAM and throughput per mode
def print_vram_throughput(df: pd.DataFrame):
    if 'peak_vram_gb' not in df.columns: return
    print(bolden(f"\nVRAM & Throughput by Mode:"))
    stats = df.groupby('mode').agg(
        avg_vram=('peak_vram_gb', 'mean'),
        max_vram=('peak_vram_gb', 'max'),
        avg_throughput=('throughput_s', 'mean'),
        max_throughput=('throughput_s', 'max')
    ).round(2)
    print(stats.to_string())



# Main
# ============================================================
def run_analysis(model_name="wikipedia_model_with_numbers"):
    df = load_log(model_name)
    if df.empty:
        print("No data found. Check the log file path and content.")
        return

    width = 44
    print(
        bolden(
        f"\n{'=' * width}\n"
        f"{'🔎  LOG INSIGHTS':^{width}}\n"
        f"{'=' * width}\n"
        )
    )

    print_highlights(df)
    print_summary(df)
    print_full_results(df)
    print_best_per_dataset(df)
    print_cell_vs_row(df)
    print_vram_throughput(df)

    print(f"\n{'=' * width}\n")


if __name__ == "__main__":
    # optionally pass model name as argument
    # e.g. python log_analysis.py f1_0.8972
    model = sys.argv[1] if len(sys.argv) > 1 else "wikipedia_model_with_numbers"
    run_analysis(model)
import pandas as pd
import io
import os
import sys


def quick_thesis_check(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    # 1. Setup Output File Path
    filename = os.path.basename(file_path).replace(".csv", "")
    output_dir = "my_tests/logs"
    os.makedirs(output_dir, exist_ok=True)
    summary_file = os.path.join(output_dir, f"{filename}_summary.csv")

    # Standard ReFinED Log Header [cite: 524, 529]
    header = "timestamp,dataset,mode,model,batch_size,total_rows_in_gt,n_samples_tested,nil_count,tp,fp,fn,accuracy,precision,recall,f1_score,peak_vram_gb,throughput_s,total_time_s"

    data_lines = [header]
    with open(file_path, 'r') as f:
        for line in f:
            if line.count(',') >= 15 and "timestamp" not in line:
                data_lines.append(line.strip())

    df = pd.read_csv(io.StringIO("\n".join(data_lines)))
    df = df.sort_values(['mode', 'dataset'])

    # 2. Redirect Print to File
    with open(summary_file, 'w') as out_f:
        # Helper to print to both console and file
        def logger(msg):
            print(msg)
            out_f.write(msg + "\n")

        logger("=" * 100)
        logger(f"THESIS DATA MIGRATION OVERVIEW: {os.path.basename(file_path)}")
        logger("=" * 100)

        # Main Table Data (Grouped by Mode) [cite: 310, 518]
        for mode in df['mode'].unique():
            logger(f"\n>>> FOR TABLE: {mode.upper()} MODE " + "-" * 60)

            subset = df[df['mode'] == mode][
                ['dataset', 'f1_score', 'precision', 'recall', 'throughput_s', 'peak_vram_gb']]

            table_str = subset.to_string(index=False, formatters={
                'f1_score': '{:,.4f}'.format,
                'precision': '{:,.4f}'.format,
                'recall': '{:,.4f}'.format,
                'throughput_s': '{:,.2f}'.format,
                'peak_vram_gb': '{:,.2f}'.format
            })
            logger(table_str)
            logger("-" * 85)

        # Average Efficiency Summary [cite: 384, 523]
        logger("\n" + "=" * 100)
        logger("AVERAGE STATS (For Table 5.5: Efficiency vs Accuracy)")
        logger("-" * 100)
        summary = df.groupby('mode')[['f1_score', 'throughput_s', 'peak_vram_gb']].mean()
        logger(summary.to_string(formatters={
            'f1_score': '{:,.4f}'.format,
            'throughput_s': '{:,.2f}'.format,
            'peak_vram_gb': '{:,.2f}'.format
        }))
        logger("=" * 100)

    print(f"\n[SUCCESS] Summary log saved to: {summary_file}")


# --- RUN ---
log_file = "my_tests/logs/new/experimental_results_wiki_2026-05-08.csv"
quick_thesis_check(log_file)
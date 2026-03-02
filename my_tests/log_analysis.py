import pandas as pd
from my_tests.utility.test_utils import get_dated_filename

#TODO
# TEST SCRIPT MADE BY LLM - This is a quick-and-dirty script to load the log file and print some insights.


def print_table_with_separators(df_table, group_col=None):
    lines = df_table.to_string(index=False).split('\n')
    header, rows = lines[0], lines[1:]
    delimiter = '-' * len(header)
    print(header)
    print(delimiter)
    if group_col:
        col_values = df_table[group_col].tolist()
        for i, (row, val) in enumerate(zip(rows, col_values)):
            print(row)
            if i < len(rows) - 1 and val != col_values[i + 1]:
                print('-' * len(header))
    else:
        for row in rows:
            print(row)
    print(delimiter)



# gets todays experiment results log_file
log_file = get_dated_filename()

# 1. Load the file without the comment filter (since it breaks dates)
df = pd.read_csv(log_file, skip_blank_lines=True, on_bad_lines='skip')

# 2. CLEANING: Keep only rows where the first column looks like a timestamp
df = df[df['timestamp'].str.contains('2026', na=False)]

# 3. CONVERSION: Ensure numeric types
cols = ['accuracy', 'precision', 'recall', 'f1_score', 'throughput_s', 'fn']
for col in cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=['accuracy'])


# --- THE ANALYSIS ---
print(f"\n{'=' * 15} 🔎 LOG INSIGHTS {'=' * 15}\n")

if not df.empty:
    best_acc = df.loc[df['accuracy'].idxmax()]
    print(f"🏆 Top Accuracy: {best_acc['accuracy']:.2%} | {best_acc['dataset']} ({best_acc['mode']})")

    fastest = df.loc[df['throughput_s'].idxmax()]
    print(f"⚡ Peak Speed:   {fastest['throughput_s']:.1f} tx/s | {fastest['dataset']} ({fastest['mode']})")

    struggler = df.loc[df['f1_score'].idxmin()]
    print(f"❌ Hardest Task: {struggler['dataset']} ({struggler['mode']}) — F1: {struggler['f1_score']:.4f}")

    print(f"\n📝 Run Summary:")
    print(f"   - Datasets Logged: {len(df)}")
    print(f"   - Average F1:      {df['f1_score'].mean():.4f}")

    # Full table with mode
    print(f"\n📋 Full Results:")
    print_table_with_separators(df[['dataset', 'mode', 'accuracy', 'f1_score', 'throughput_s']], group_col='mode')

    # Best mode per dataset
    print(f"\n📊 Best F1 per dataset:")
    best_per_dataset = df.loc[df.groupby('dataset')['f1_score'].idxmax()][['dataset', 'mode', 'accuracy', 'f1_score']]
    print_table_with_separators(best_per_dataset.sort_values('f1_score', ascending=False).reset_index(drop=True))

else:
    print("Still no data. Double-check the path and file content.")

print(f"\n{'=' * 36}")
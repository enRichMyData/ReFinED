import pandas as pd
from my_tests.utility.test_utils import get_dated_filename


# gets todays experiment results
log_file = get_dated_filename()



# TEST SCRIPT MADE BY LLM - This is a quick-and-dirty script to load the log file and print some insights.


# 1. Load the file without the comment filter (since it breaks dates)
df = pd.read_csv(log_file, skip_blank_lines=True, on_bad_lines='skip')

# 2. CLEANING: Keep only rows where the first column looks like a timestamp
# (This automatically nukes the "--- Starting Mode ---" lines)
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
    print(f"🏆 Top Accuracy: {best_acc['accuracy']:.2%} | {best_acc['dataset']}")

    fastest = df.loc[df['throughput_s'].idxmax()]
    print(f"⚡ Peak Speed:   {fastest['throughput_s']:.1f} tx/s | {fastest['dataset']}")

    struggler = df.loc[df['fn'].idxmax()]
    print(f"❌ Hardest Task: {struggler['dataset']} (Missed {int(struggler['fn'])} rows)")

    print(f"\n📝 Run Summary:")
    print(f"   - Datasets Logged: {len(df)}")
    print(f"   - Average F1:      {df['f1_score'].mean():.4f}")

    # Let's see the raw table for a second
    print(f"\n{df[['dataset', 'accuracy', 'f1_score', 'throughput_s']].to_string(index=False)}")
else:
    print("Still no data! Double-check the path and file content.")

print(f"\n{'=' * 36}")
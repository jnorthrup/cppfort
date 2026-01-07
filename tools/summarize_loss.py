import csv
import sys

csv_file = sys.argv[1]

total_loss = 0.0
count = 0
high_loss_count = 0
failures = 0
skips = 0
perfect = 0

print(f"{'File':<60} {'Loss':<10} {'Status':<10}")
print("-" * 80)

with open(csv_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        status = row['status']
        if status == 'OK':
            loss = float(row['loss'])
            total_loss += loss
            count += 1
            if loss > 0.15:
                print(f"{row['file']:<60} {loss:.3f}      HIGH")
                high_loss_count += 1
            elif loss == 0.0:
                print(f"{row['file']:<60} {loss:.3f}      PERFECT")
                perfect += 1
        elif status.startswith('SKIP'):
            skips += 1
        else:
            failures += 1
            print(f"{row['file']:<60} -          {status}")

print("-" * 80)
if count > 0:
    avg_loss = total_loss / count
    print(f"Average Loss: {avg_loss:.4f}")
else:
    print("Average Loss: N/A")

print(f"Total Files: {count + failures + skips}")
print(f"Scored: {count}")
print(f"Perfect (0.0): {perfect}")
print(f"High Loss (>0.15): {high_loss_count}")
print(f"Failures: {failures}")
print(f"Skips: {skips}")

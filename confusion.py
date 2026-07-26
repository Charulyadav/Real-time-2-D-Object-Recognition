  #Charul - 002535330
  #Spring 2026
  #CS 5330 Computer Vision
  #Project 3: Real-time 2-D Object Recognition

  #to build confusion matrix for Task 7

import csv
from collections import defaultdict

pairs = []
with open("eval_results.csv") as f:
    for row in csv.reader(f):
        if len(row) == 2:
            pairs.append((row[0].strip(), row[1].strip()))

labels = sorted(set([t for t, p in pairs] + [p for t, p in pairs]))
matrix = defaultdict(lambda: defaultdict(int))
for true, pred in pairs:
    matrix[true][pred] += 1

print("\nConfusion Matrix (rows = true, cols = predicted)\n")
print("true\\pred".ljust(12) + "".join(l.ljust(12) for l in labels))
for true in labels:
    row = true.ljust(12)
    for pred in labels:
        row += str(matrix[true][pred]).ljust(12)
    print(row)

correct = sum(1 for t, p in pairs if t == p)
print(f"\nAccuracy: {correct}/{len(pairs)} = {100*correct/len(pairs):.1f}%")
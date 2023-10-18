import glob
import sys

assert len(sys.argv) == 2, "Usage: python check_cv_loss.py <exp_dir>"
exp_dir = sys.argv[1]

cv_logs = sorted(glob.glob(f"exp/{exp_dir}/*.yaml"))[:-2]

cv_logs.sort(key=lambda x: int(x.split("/")[-1].split(".")[0]))

for i, log in enumerate(cv_logs):
    with open(log, "r") as f:
        lines = f.readlines()
        print(f"Epoch {i}, cv_loss {float(lines[0].split()[-1]):.4f}")

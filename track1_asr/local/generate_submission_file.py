import os
import sys


if __name__ == '__main__':
    assert len(sys.argv) == 2, "Usage: python3 generate_submission_file.py <test_dir>"
    test_dir = sys.argv[1]
    text_file = f"{test_dir}/text"
    if not os.path.exists(text_file):
        print(f"Error: {text_file} not found. Please check the path.")
        exit(1)
    print(f"Generating submission file for {test_dir} ...")
    lines = []
    for line in open(text_file, 'r').readlines():
        line_splits = line.split()
        if len(line_splits) != 2:
            # blank decoding results
            lines.append([line_splits[0], ''])
        else:
            lines.append([line_splits[0], line_splits[1].strip()])

    # Hide the channel information in utt id
    for i, line in enumerate(lines):
        utt_id_splits = line[0].split('_')
        utt_id = '_'.join(utt_id_splits[:2] + utt_id_splits[3:])
        lines[i][0] = utt_id
    lines = sorted(lines, key=lambda x: x[0])
    with open(f"{test_dir}/submission.txt", 'w') as f:
        for line in lines:
            f.write(f"{line[0]} {line[1]}\n")

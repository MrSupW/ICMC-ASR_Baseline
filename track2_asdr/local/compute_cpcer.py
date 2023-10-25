#!/usr/bin/env python
# -- coding: UTF-8

import argparse
import codecs
import logging
import editdistance
import numpy as np

from scipy.optimize import linear_sum_assignment


def get_parser():
    parser = argparse.ArgumentParser(description="Compute cpCER.")
    parser.add_argument(
        "--hyp-path", type=str, required=True, help="File path of decoded results."
    )
    parser.add_argument(
        "--ref-path", type=str, required=True, help="File path of ground truth results."
    )

    return parser


def read_hyp_and_ref(hyp_path, ref_path):
    content_hyp, content_ref = [], []
    # read hyp file
    with codecs.open(hyp_path, "r") as f_hyp:
        for line in f_hyp.readlines():
            line = line.replace("<sos/eos>", "").strip()
            if len(line.split(" ")) == 1:
                # remove empty results
                continue
            content_hyp.append(line)
    # read ref file
    with codecs.open(ref_path, "r") as f_ref:
        for line in f_ref.readlines():
            line = line.strip()
            content_ref.append(line)

    output_from_hyp = []
    output_from_ref = []
    # format hyp content
    for line in content_hyp:
        file_name = "_".join(line.split(" ")[0].split("_")[:-1])
        if len(output_from_hyp) == 0 or output_from_hyp[-1].split(" ")[0] != file_name:
            output_from_hyp.append(" ".join([file_name] + line.split(" ")[1:]))
        else:
            output_from_hyp[-1] += " ".join(line.split(" ")[1:])
    # format ref content
    for line in content_ref:
        file_name = "_".join(line.split(" ")[0].split("_")[:-1])
        if len(output_from_ref) == 0 or output_from_ref[-1].split(" ")[0] != file_name:
            output_from_ref.append(" ".join([file_name] + line.split(" ")[1:]))
        else:
            output_from_ref[-1] += " ".join(line.split(" ")[1:])

    return output_from_hyp, output_from_ref


def generate_hyp_and_ref_dict(hyp_content, ref_content):
    # generate dict for hyp content
    hyp_lines = sorted(hyp_content)
    hyp_dict = {}
    for line in hyp_lines:
        key, value = line.split(" ")
        uttid = "_".join(key.split("_")[1:])
        spkid = key.split("_")[0]
        if uttid not in hyp_dict.keys():
            hyp_dict[uttid] = []
        hyp_dict[uttid].append((spkid, value))

    # generate dict for ref_content
    ref_lines = sorted(ref_content)
    ref_dict = {}
    for line in ref_lines:
        key, value = line.split(" ")
        uttid = "_".join(key.split("_")[1:])
        spkid = key.split("_")[0]
        if uttid not in ref_dict.keys():
            ref_dict[uttid] = []
        ref_dict[uttid].append((spkid, value))

    return hyp_dict, ref_dict


def compute_cer(hyp_seq, ref_seq):
    hyp_seq = hyp_seq.replace("<unk>", "*")

    hyp = list(hyp_seq)
    len_hyp = len(hyp)
    ref = list(ref_seq)
    len_ref = len(ref)

    cost_matrix = np.zeros((len_hyp + 1, len_ref + 1), dtype=np.int16)
    ops_matrix = np.zeros((len_hyp + 1, len_ref + 1), dtype=np.int8)

    for i in range(len_hyp + 1):
        cost_matrix[i][0] = i
    for j in range(len_ref + 1):
        cost_matrix[0][j] = j

    for i in range(1, len_hyp + 1):
        for j in range(1, len_ref + 1):
            if hyp[i - 1] == ref[j - 1]:
                cost_matrix[i][j] = cost_matrix[i - 1][j - 1]
            else:
                substitution = cost_matrix[i - 1][j - 1] + 1
                insertion = cost_matrix[i - 1][j] + 1
                deletion = cost_matrix[i][j - 1] + 1
                compare_val = [substitution, insertion, deletion]

                min_val = min(compare_val)
                operation_idx = compare_val.index(min_val) + 1
                cost_matrix[i][j] = min_val
                ops_matrix[i][j] = operation_idx

    match_idx = []
    i = len_hyp
    j = len_ref
    nb_map = {"N": len_ref, "C": 0, "W": 0, "I": 0, "D": 0, "S": 0}
    while i >= 0 or j >= 0:
        i_idx = max(0, i)
        j_idx = max(0, j)

        if ops_matrix[i_idx][j_idx] == 0:  # correct
            if i - 1 >= 0 and j - 1 >= 0:
                match_idx.append((j - 1, i - 1))
                nb_map["C"] += 1

            i -= 1
            j -= 1
        elif ops_matrix[i_idx][j_idx] == 2:  # insert
            i -= 1
            nb_map["I"] += 1
        elif ops_matrix[i_idx][j_idx] == 3:  # delete
            j -= 1
            nb_map["D"] += 1
        elif ops_matrix[i_idx][j_idx] == 1:  # substitute
            i -= 1
            j -= 1
            nb_map["S"] += 1

        if i < 0 and j >= 0:
            nb_map["D"] += 1
        elif j < 0 and i >= 0:
            nb_map["I"] += 1

    match_idx.reverse()
    wrong_cnt = cost_matrix[len_hyp][len_ref]
    nb_map["W"] = wrong_cnt

    return nb_map["N"], nb_map["I"], nb_map["D"], nb_map["S"]


def compute_cpCER(hyp_content, ref_content):
    assert len(hyp_content) == len(ref_content)
    length = len(hyp_content)
    cost_matrix = np.zeros((length, length))

    for i in range(length):
        for j in range(length):
            cost = editdistance.eval(hyp_content[i][1], ref_content[j][1])
            cost_matrix[i][j] = cost

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # mapping speaker id of rttm to global speaker id
    assert len(row_ind) == len(col_ind)
    for i in range(len(row_ind)):
        print(
            "Hyp spkr id: {} -> Ref spkr id: {}".format(
                hyp_content[row_ind[i]][0], ref_content[col_ind[i]][0]
            )
        )
    cost = cost_matrix[row_ind, col_ind].sum()

    entire = 0
    for i in range(length):
        entire += len(list(ref_content[i][1]))

    return cost, entire


def main():
    args = get_parser().parse_args()

    # logging info
    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=logfmt)

    output_from_hyp, output_from_ref = read_hyp_and_ref(args.hyp_path, args.ref_path)
    hyp_dict, ref_dict = generate_hyp_and_ref_dict(output_from_hyp, output_from_ref)
    hyp_sessions, ref_sessions = list(hyp_dict.keys()), list(ref_dict.keys())
    assert len(hyp_sessions) == len(
        ref_sessions
    ), "The number of sessions doesn't match: {} vs {}".format(
        hyp_sessions, ref_sessions
    )

    cost_total, entire_total = 0, 0
    for idx, sname in enumerate(hyp_sessions):
        hyp_content = hyp_dict[sname]
        if sname not in ref_sessions:
            logging.warning("Missing session {} in reference.".format(sname))
            continue
        ref_content = ref_dict[sname]

        if len(hyp_content) != len(ref_content):
            logging.warning(
                "Missing speakers between hyp: {} vs ref: {}".format(
                    len(hyp_content), len(ref_content)
                )
            )
            max_len = max(len(hyp_content), len(ref_content))
            if len(hyp_content) != max_len:
                for i in range(max_len - len(hyp_content)):
                    hyp_content.append(("SNAN", ""))
            else:
                for i in range(max_len - len(ref_content)):
                    ref_content.append(("SNAN", ""))

        print("Session Name: {}".format(sname))
        cost, entire = compute_cpCER(hyp_content, ref_content)
        logging.info(
            "Computing: {}/{}, cpCER is: {:.4f}".format(
                idx + 1, len(hyp_sessions), cost / entire
            )
        )
        cost_total += cost
        entire_total += entire

    cpCER = cost_total / entire_total
    logging.info("Total cpCER is: {:.4f}".format(cpCER))


if __name__ == "__main__":
    main()

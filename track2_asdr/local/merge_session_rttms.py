import argparse
import glob
import os
import numpy as np


def main(args):
    rttm = {}
    rttm2dur = {}

    if os.path.exists(args.save_path):
        os.system(f"rm {args.save_path}")

    for rttm_file in glob.glob(os.path.join(args.segments_path, "*.rttm"), recursive=False):
        session = rttm_file.split("/")[-1].split(".")[0]
        with open(rttm_file, "r") as f:
            for line in f.readlines():
                start = line.strip().split()[3]
                dur = line.strip().split()[4]
                start, dur = float(start), float(dur)
                rttm.setdefault(session, []).append((round(start, 2), round(start + dur, 2)))
                if session not in rttm2dur:
                    rttm2dur[session] = 0.
                else:
                    rttm2dur[session] += dur

    for utt in rttm.keys():
        rttm[utt].sort()

    result = {}
    result_save_index = {}

    for session in sorted(rttm.keys()):
        segs = rttm[session]
        real_session, channel = session.split("_")
        if real_session not in result:
            result[real_session] = []
            durs = []
            channel = channel[:3] + "1" + channel[4:]
            result[real_session].append(rttm[real_session + "_" + channel])
            durs.append(rttm2dur[real_session + "_" + channel])

            channel = channel[:3] + "2" + channel[4:]
            result[real_session].append(rttm[real_session + "_" + channel])
            durs.append(rttm2dur[real_session + "_" + channel])

            channel = channel[:3] + "3" + channel[4:]
            result[real_session].append(rttm[real_session + "_" + channel])
            durs.append(rttm2dur[real_session + "_" + channel])

            channel = channel[:3] + "4" + channel[4:]
            result[real_session].append(rttm[real_session + "_" + channel])
            durs.append(rttm2dur[real_session + "_" + channel])

            # only two speakers in each session. so we only need to save the two longest duration speakers
            result_save_index[real_session] = np.argsort(durs)[-2:]

    with open(args.save_path, "w") as wf:
        for session, segs in result.items():
            for i in range(len(segs)):
                if i in result_save_index[session]:
                    for seg in segs[i]:
                        print(f"SPEAKER {session} 1 {seg[0]} {round(seg[1] - seg[0], 2)} <NA> <NA> {i + 1} <NA> <NA>",
                              file=wf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    main(parser.parse_args())

import os
import sys
import glob
import argparse
import numpy as np
import soundfile as sf

sys.path.append("./enhancement")
from tqdm import tqdm
from enhancement.iva import iva
from enhancement.pfdkf import pfdkf


def main(args):
    session2utt = {}

    with open(args.wav_scp, "r") as f:
        for line in f.readlines():
            utt, path = line.strip().split()
            session2utt.setdefault(utt.split("_")[0], []).append(path)

    sr = 16000
    for session, utts in session2utt.items():
        utts.sort()
        os.makedirs(os.path.join(args.save_path, session), exist_ok=True)
        wavs = [sf.read(utt)[0] for utt in utts]
        # Do AEC First
        session_dir = os.path.dirname(utts[0])
        ref_utts = glob.glob(os.path.join(session_dir, "DX0[5-6]C01.wav"))
        if len(ref_utts) != 0:
            # If there is reference, do AEC
            print(f"Doing AEC for session {session}")
            ref_utts.sort()
            ref1, _ = sf.read(ref_utts[0])
            ref2, _ = sf.read(ref_utts[1])
            for i in range(len(wavs)):
                mic = wavs[i]
                error1, echo1 = pfdkf(ref1, mic, A=0.999, keep_m_gate=0.5)
                error2, echo2 = pfdkf(ref2, mic, A=0.999, keep_m_gate=0.5)
                echo = (echo1 + echo2) / 2.0
                min_len = min(len(echo), len(mic))
                mic = mic[:min_len]
                echo = echo[:min_len]
                wavs[i] = mic - echo
        else:
            print(f"No reference for session {session}, skip AEC")

        print(f"Doing IVA for session {session}")
        min_length = min([wav.shape[0] for wav in wavs])
        wavs = [wav[:min_length] for wav in wavs]
        x = np.stack(wavs, axis=1)
        y = iva(x)

        for i in range(y.shape[0]):
            sf.write(os.path.join(args.save_path, session, f"DX0{i + 1}C01.wav"), y[i], sr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_scp", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    main(parser.parse_args())

import os
import torch
import argparse
from tqdm import tqdm

from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection


def main(args):
    if os.path.exists(args.save_path):
        os.system(f"rm -rf {args.save_path}")
    os.makedirs(args.save_path)

    wav_scp = {}
    with open(args.wav_scp, "r") as f:
        for line in f.readlines():
            utt, path = line.strip().split()
            wav_scp[utt] = path

    model = Model.from_pretrained("pyannote/segmentation", use_auth_token=args.token).cuda()
    pipeline = VoiceActivityDetection(segmentation=model)

    HYPER_PARAMETERS = {
        # onset/offset activation thresholds
        "onset": args.threshold, "offset": 0.5,
        # remove speech regions shorter than that many seconds.
        "min_duration_on": 0.1,
        # fill non-speech regions shorter than that many seconds.
        "min_duration_off": 0.3
    }

    pipeline.instantiate(HYPER_PARAMETERS)
    for utt, path in tqdm(wav_scp.items()):
        vad = pipeline(path)
        with open(os.path.join(args.save_path, f"{utt}.rttm"), "w") as rttm:
            vad.write_rttm(rttm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav_scp', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--threshold', type=float, required=True)
    parser.add_argument('--token', type=str, required=True)

    main(parser.parse_args())

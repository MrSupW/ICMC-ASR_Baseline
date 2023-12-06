import os
import glob
import sys

sys.path.append("./normalize")

from tqdm import tqdm
from normalize.cn_tn import TextNorm


def prepare_data(dataset_pf):
    # train dev eval_track1
    audio_dirs = glob.glob(f"{data_root}/{dataset_pf}/*")
    wav_scp = []
    text = []
    for audio_dir in tqdm(audio_dirs):
        # near field audio
        near_textgrids = glob.glob(f"{audio_dir}/DA0*.TextGrid")
        # 1 2 3 4
        near_seat_ids = [tg.split('/')[-1][3] for tg in near_textgrids]
        near_wav_dirs = [tg.replace('.TextGrid', '') for tg in near_textgrids]
        # search far-field audio by seat id
        far_wav_dirs = [f"{audio_dir.replace(data_root, enhanced_data_root)}/DX0{seat_id}C01"
                        for seat_id in near_seat_ids]
        txt_files = glob.glob(f"{audio_dir}/*.txt")
        if dataset_pf == "train":
            for dir_ in near_wav_dirs + far_wav_dirs:
                wav_scp.extend([(f"{file.split('/')[-1].replace('.wav', '')}", file)
                                for file in glob.glob(f"{dir_}/*.wav")])
        else:
            for dir_ in far_wav_dirs:
                wav_scp.extend([(f"{file.split('/')[-1].replace('.wav', '')}", file)
                                for file in glob.glob(f"{dir_}/*.wav")])

        for txt_file in txt_files:
            lines = open(txt_file).readlines()
            text.extend([(line.split()[0],
                          text_normalizer(line.split()[1].strip().replace('﻿', '')).replace('2', '二'))
                         for line in lines])

    wav_scp.sort(key=lambda x: x[0])
    text.sort(key=lambda x: x[0])
    assert len(wav_scp) == len(text), f"wav_scp: {len(wav_scp)}, text: {len(text)}"
    for i in range(len(wav_scp)):
        assert wav_scp[i][0] == text[i][0]

    os.system(f"mkdir -p ./data/{dataset}")
    with open(f"./data/{dataset}/wav.scp", "w") as f:
        for line in wav_scp:
            f.write(f"{line[0]} {line[1]}\n")
    with open(f"./data/{dataset}/text", "w") as f:
        for line in text:
            f.write(f"{line[0]} {line[1]}\n")
    with open(f"./data/{dataset}/utt2spk", "w") as f:
        for line in wav_scp:
            f.write(f"{line[0]} {line[0].split('_')[0]}\n")

    os.system(f"./tools/utt2spk_to_spk2utt.pl ./data/{dataset}/utt2spk > ./data/{dataset}/spk2utt")
    os.system(f"./tools/fix_data_dir.sh ./data/{dataset}")
    os.system(f"./tools/validate_data_dir.sh --no-feats ./data/{dataset}")


if __name__ == '__main__':
    data_root, enhanced_data_root, dataset = sys.argv[1], sys.argv[2], sys.argv[3]
    dataset_prefix = dataset.split('_aec_iva')[0]
    text_normalizer = TextNorm(to_banjiao=True, to_upper=True)
    prepare_data(dataset_prefix)

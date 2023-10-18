<h1 align="center">ICASSP2024 ICMC-ASR Challenge Baseline</h1>

## Introduction

This repository is the baseline code for the ICMC-ASR (In-Car Multi-Channel Automatic Speech Recognition).

The code in this repository is based on the End-to-End Speech Recognition Toolkit [WeNet]([wenet-e2e/wenet: Production First and Production Ready End-to-End Speech Recognition Toolkit (github.com)](https://github.com/wenet-e2e/wenet)) and Speaker Recognition Toolkit [WeSpeaker]([wenet-e2e/wespeaker: Research and Production Oriented Speaker Recognition Toolkit (github.com)](https://github.com/wenet-e2e/wespeaker)).

## Data Preparation

Before running this baseline, you should have downloaded and unzipped the dataset for this challenge, whose folder structure is as follows:

```Shell
ICMC-ASR
└── train
|   ├── 001
|   ├── 002
|   ├── 003
|   |   ├── DA0*.wav (near-field headset audios)
|   |   ├── DA0*.TextGrid (label files including speaker label, timestamps and text content)
|   |   ├── DX0[1-4]C01 (4-channel far-field audios)
|   |   └── DX0[5-6]C01 (2-channel reference signals for AEC only existing when the car stereo is turned on)
|   ├── ...
|   └── 568
├── dev
|   ├── 001
|   ├── 002
|   |   ├── DA0*.wav (near-field headset audios)
|   |   ├── DA0*.TextGrid (label files including speaker label, timestamps and text content)
|   |   ├── DX0[1-4]C01 (4-channel far-field audios)
|   |   └── DX0[5-6]C01 (2-channel reference signals for AEC only existing when the car stereo is turned on)
|   ├── ...
|   ├── 018
├── eval_track1 (waiting to release)
|   ├── 001
|   ├── 002
|   |   ├── DA0*.TextGrid (label files including speaker label, timestamps without text content)
|   |   ├── DX0[1-4]C01 (4-channel far-field audios)
|   |   └── DX0[5-6]C01 (2-channel reference signals for AEC only exist when the car stereo is turned on)
|   ├── ...
|   ├── 018
└── eval_track2 (waiting to release)
    ├── 001
    ├── 002
    |   ├── DX0[1-4]C01 (4-channel far-field audios)
    |   └── DX0[5-6]C01 (2-channel reference signals for AEC only exist when the car stereo is turned on)
    ├── ...
    └── 018
```

### Notice:fire: 

We have released the latest near-field audio and textgrids of the training and development sets and fixed the issue with incorrect speaker seat (near-field audio and textgrid's file name) information that was reported by some teams. We strongly recommend you to get them from the download links in the email and replace **all** near-field audio and textgrids from the previously downloaded training and development sets.

## Environment Setup

```Shell
# create environment
conda create -n icmcasr python=3.9 -y
conda activate icmcasr
# install cudatoolkit and cudnn
conda install cudatoolkit=11.6 cudnn=8.4.1 -y
# install pytorch torchvision and torchaudio
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
# install other dependence
pip install -r requirements.txt
```

## Track1 Baseline

The main steps are all in track1_asr/run.sh

**[Stage 0]** Audio Frontend IVA + AEC. Segment long waves into short slices and prepare the data files (wav.scp, text, utt2spk and spk2utt).

**[Stage 1]** Compute CMVN value of training set for audio feature normalization.

**[Stage 2]** Generate the token list (char).

**[Stage 3]** Prepare the data in the WeNet required format.

**[Stage 4]** Train the ASR model following the train config file.

**[Stage 5]** Do model average and decoding.

## Track2  Baseline (Coming soon)

Before running the track2 baseline, please make sure you have run all the stages in track1_asr/run.sh and get the trained ASR model.


## License

It is noted that the code can only be used for comparative or benchmarking purposes. Users can only use code supplied under a [License](./LICENSE) for non-commercial purposes.

## Contact

```
[ICMC-ASR Committee](icmcasr_challenge@aishelldata.com)
```


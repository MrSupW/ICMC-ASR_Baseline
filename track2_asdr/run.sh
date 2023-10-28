#!/bin/bash

# Copyright 2019 Mobvoi Inc. All Rights Reserved.
. ./path.sh || exit 1;

# Use this to control how many gpu you use, It's 1-gpu training if you specify
# just 1gpu, otherwise it's is multiple gpu training based on DDP in pytorch
export CUDA_VISIBLE_DEVICES="0,1,2,3"
# for debug purpose, please set it to 1 otherwise, set it to 0
export CUDA_LAUNCH_BLOCKING=0

stage=0 # start from 0 if you need to start from data preparation
stop_stage=5

# Create your access token at https://huggingface.co/settings/tokens
HUGGINGFACE_ACCESS_TOKEN="YOUR_HUGGINGFACE_ACCESS_TOKEN"
################################################
# The icmc-asr dataset location, please change this to your own path!!!
# Make sure of using absolute path. DO-NOT-USE relatvie path!
# data dir for IVA + AEC enhanced audio
data_enhanced=/home/work_nfs4_ssd/hwang/data/ICMC-ASR_ENHANCED
################################################

nj=64
dict=data/dict/lang_char.txt

# Pyannote VAD activation threshold
threshold=0.95
# data_type can be `raw` or `shard`. Typically, raw is used for small dataset,
# `shard` is used for large dataset which is over 1k hours, and `shard` is
# faster on reading data and training.
data_type=raw
num_utts_per_shard=1000

test_set=dev_aec_iva
dir=exp/baseline_ebranchformer

# use average_checkpoint will get better result
decode_checkpoint=$dir/avg_10.pt
decode_modes="ctc_greedy_search ctc_prefix_beam_search attention attention_rescoring"

. tools/parse_options.sh || exit 1;


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "stage 0: Do VAD for enhanced audio data"
  for x in ${test_set} ; do
    # enhanced_data_root dataset threshold HUGGINGFACE_ACCESS_TOKEN
    local/run_pyannote_vad.sh $data_enhanced $x $threshold $HUGGINGFACE_ACCESS_TOKEN
  done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "stage 1: Segment audio data based on VAD results"
  for x in ${test_set} ; do
    python3 local/segment_wavs_by_rttm.py --enhanced_data_root $data_enhanced --dataset $x --nj $nj \
     exp/pyannote_vad/${x}_${threshold}/pyannote_vad.rttm data/${x}_${threshold}/pyannote_vad_wavs/
  done
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "stage 2: Prepare data in WeNet required format"
  for x in ${test_set}; do
    if [ $data_type == "shard" ]; then
      tools/make_shard_list.py --num_utts_per_shard $num_utts_per_shard \
        --num_threads ${nj} data/${x}_${threshold}/wav.scp data/$x/text \
        $(realpath data/${x}_${threshold}/shards) data/${x}_${threshold}/data.list
    else
      tools/make_raw_list.py data/${x}_${threshold}/wav.scp data/${x}_${threshold}/text \
        data/${x}_${threshold}/data.list
    fi
  done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  # Test model, please specify the model you want to test by --checkpoint
  echo "stage 3: Test model testset ${test_set}_${threshold}"
  mkdir -p $dir

  if [ ! -f $decode_checkpoint ]; then
    echo "error: $decode_checkpoint does not exist."
    echo "please copy the trained model from track1 to $decode_checkpoint"
    exit 1
  fi

  num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
  # Please specify decoding_chunk_size for unified streaming and
  # non-streaming model. The default value is -1, which is full chunk
  # for non-streaming inference.
  decoding_chunk_size=
  ctc_weight=0.3
  idx=0
  for mode in ${decode_modes}; do
  {
    test_dir=$dir/${test_set}_${mode}_${threshold}
    mkdir -p $test_dir
    gpu_id=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f$[$idx+1])
    python3 wenet/bin/recognize.py --gpu $gpu_id \
      --mode $mode \
      --config $dir/train.yaml \
      --data_type $data_type \
      --test_data data/${test_set}_${threshold}/data.list \
      --checkpoint $decode_checkpoint \
      --beam_size 10 \
      --batch_size 1 \
      --penalty 0.0 \
      --dict $dict \
      --ctc_weight $ctc_weight \
      --result_file $test_dir/text \
      ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size}
  } &
  ((idx+=1))
  if [ $idx -eq $num_gpus ]; then
    idx=0
  fi
  done
  wait
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "stage 4: Generate submission file for track2 leaderboard"
  for mode in ${decode_modes}; do
  {
    test_dir=$dir/${test_set}_${mode}_${threshold}
    python3 local/generate_submission_file.py "$test_dir"
  }
  done
fi

# only for dev_aec_iva set
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "stage 5: Compute cpCER of dev_aec_iva set"
  for mode in ${decode_modes}; do
  {
    echo "compute cpCER for ${test_set}_${mode}_${threshold}"
    test_dir=$dir/${test_set}_${mode}_${threshold}
    python3 local/compute_cpcer.py --hyp-path $test_dir/submission.txt --ref-path data/dev_ref.txt
    echo ""
  }
  done
fi

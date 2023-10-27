. ./path.sh || exit 1

stage=0
stop_stage=2
nj=48

. tools/parse_options.sh

data_root=$1
enhanced_data_root=$2
dataset=$3


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "[local/icmcasr_data_prep.sh] stage 0: AEC + IVA Enhancement"
  local/enhancement.sh --nj ${nj} ${data_root} ${enhanced_data_root} ${dataset}
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "[local/icmcasr_data_prep.sh] stage 1: Segment ${dataset} wavs"
  python3 local/segment_wavs.py ${data_root} ${enhanced_data_root} ${dataset} ${nj}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "[local/icmcasr_data_prep.sh] stage 2: Prepare data files"
  python3 local/data_prep.py ${data_root} ${enhanced_data_root} ${dataset}
fi

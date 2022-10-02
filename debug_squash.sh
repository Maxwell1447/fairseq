#!/bin/bash

INFER_SCRIPTS_PATH=/linkhome/rech/genrqo01/ufn16wp/NLP4NLP/scripts/multi-lev/scripts/infer
DATA_GEN_PATH=/linkhome/rech/genrqo01/ufn16wp/NLP4NLP/DATA/multi-domain/generated

source /gpfswork/rech/usb/ufn16wp/miniconda3/bin/activate fairseq

cd $INFER_SCRIPTS_PATH
source $INFER_SCRIPTS_PATH/run_sbatch_infer_lev_debug.sh

# cd $DATA_GEN_PATH
# source $DATA_GEN_PATH/run_eval.sh "-unsquashed"
# source $DATA_GEN_PATH/run_eval.sh "-squashed"

##################################################################

# cd $INFER_SCRIPTS_PATH
# source $INFER_SCRIPTS_PATH/run_sbatch_infer_lev_debug.sh


# cd /linkhome/rech/genrqo01/ufn16wp/NLP4NLP/fairseq
# python debug_squash.py

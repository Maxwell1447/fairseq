# Towards Example-Based NMT with Multi-Levenshtein Transformers

## Introduction

This work is based on [fairseq](https://github.com/facebookresearch/fairseq). 
See original [fairseq README](README-fairseq.md) for more information about fairseq.
It is an extension of the code of [Levenstein Transformer](https://arxiv.org/abs/1905.11006) to mainly adapt it to multiple sentences.

Find our paper here: [Towards Example-Based NMT with Multi-Levenshtein Transformers](https://arxiv.org/abs/2310.08967)

Abstract:
> Retrieval-Augmented Machine Translation
(RAMT) is attracting growing attention. This
is because RAMT not only improves transla-
tion metrics, but is also assumed to implement
some form of domain adaptation. In this contri-
bution, we study another salient trait of RAMT,
its ability to make translation decisions more
transparent by allowing users to go back to ex-
amples that contributed to these decisions. For
this, we propose a novel MT architecture aim-
ing to increase this transparency. This model
adapts a retrieval-augmented version of the Lev-
enshtein Transformer and makes it amenable
to simultaneously edit multiple fuzzy matches
found in memory. We discuss how to perform
training and inference in this model, based on
multi-way alignment algorithms and imitation
learning. Our experiments show that editing
several examples positively impacts translation
scores, notably increasing the number of target
spans that are copied from existing instances.

## Code changed

* [`setup.py`](./setup.py)
* [`fairseq_cli/generate.py`](./fairseq_cli/generate.py) $\longrightarrow$ to integrate precision scores + HTML formatted output
* [`fairseq/clib/libnat2`](./fairseq/clib/libnat2) $\longrightarrow$ C++ DP alignment algorithm
* [`fairseq/clib/dist_realign_cuda`](./fairseq/clib/dist_realign_cuda) $\longrightarrow$ CUDA/C++ realignment gradient descent
* `fairseq/data`:
  * [`multi_source_dataset.py`](./multi_source_dataset.py) $\longrightarrow$ dataset to handle multiple examples additionally to the source and target.
  * [`data_utils.py`](./data_utils.py) $\longrightarrow$ added filter to remove unfitting samples
* `fairseq/dataclass/configs` $\longrightarrow$ added new custom parameters
* `fairseq/tasks/`:
  * [`translation_lev.py`](./translation_lev.py) $\longrightarrow$ to handle TMs in LevT
  * [`translation_multi_lev.py`](./translation_multi_lev.py) $\longrightarrow$ new task class for mLevT
* `fairseq/models/nat`:
  * [`levenshtein_transformer.py`](./levenshtein_transformer.py) $\longrightarrow$ added code to track origin of hypothesis tokens in LevT
  * [`levenshtein_utils.py`](./levenshtein_utils.py) $\longrightarrow$ added code to track origin of hypothesis tokens in LevT
  * [`multi_levenshtein_transformer.py`](./multi_levenshtein_transformer.py) $\longrightarrow$ new model class for mLevT
  * [`multi_levenshtein_utils.py`](./multi_levenshtein_utils.py) $\longrightarrow$ set of utility functions for mLevT

## Experimental data

The multidomain data as well as the train/valid/test-0.4/test-0.6 split is the same as in (Xu et. al, 2023) [(see this github repo)](https://github.com/jitao-xu/tm-levt#data-preprocessing).

The Fuzzy Matching is performed with [Systran's open source code](https://github.com/SYSTRAN/fuzzy-match). The version used can be found in commit [f4c1d7f](https://github.com/SYSTRAN/fuzzy-match/commit/f4c1d7f), accessible via `git checkout f4c1d7f`. After compilation, the following script can be adapted to compute the Fuzzy Matches:

``` bash
FUZZY_CLI=path/to/cli/src/FuzzyMatch-cli # to adapt
DATA=path/to/data # to adapt
l1=... # to adapt
l2=... # to adapt

# Indexing
$FUZZY_CLI -a index -c $DATA.$l1,$DATA.$l2 --add-target
# generated $DATA.$l1.fmi

# Matching
$FUZZY_CLI -i $DATA.$l1.fmi -a match \
-f 0.4 \
-N 8 \
-n 3 \
--ml 3 \
--mr 0.3 \
--no-perfect \
< $DATA.en > $DATA.$l1.match
```

The generated `$DATA.$l1.match` must then be processed to generate `n` files: `$DATA.$l2{1,2,3}` for `n=3`, where each line is a similar sentence (or an empty line if none retrieved). This this the expected format fore preprocessing.


## Data preprocessing

Not only the $(\textbf{x}, \textbf{y})$ source/target pairs need to be processed but also the retrieved examples $(\textbf{y}_1, \cdots, \textbf{y}_N)$. There are two steps: Tokenization+BPE and Binarization


* **Tokenization+BPE**

This step has to be done for each dataset and data split (train, valid, tests...).
It requires packages [Moses](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/tokenizer.perl) and [Subword-NMT](https://github.com/rsennrich/subword-nmt).
The "..." are to be filled by the user accordingly. 

``` bash
PATH_TO_DATA_RAW=...
PATH_TO_DATA_BPE=...
l1=...
l2=...
N=...
BPE_CODE=...
SCRIPTS=.../mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
DETOKENIZER=$SCRIPTS/tokenizer/detokenizer.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
BPEROOT=.../subword-nmt/subword_nmt

# source + target
for lang in $l1 $l2
do
  cat $PATH_TO_DATA_RAW.$lang | \
      perl $NORM_PUNC $lang | \
      perl $REM_NON_PRINT_CHAR | \
      perl $TOKENIZER -threads 40 -a -l $lang | \
      python $BPEROOT/apply_bpe.py -c $BPE_CODE > $PATH_TO_DATA_BPE.$lang
done;
# N examples
for i in $(seq 1 $N)
do
    cat $PATH_TO_DATA_RAW.$l2$i | \
        perl $NORM_PUNC $l2 | \
        perl $REM_NON_PRINT_CHAR | \
        perl $TOKENIZER -threads 40 -a -l $l2 | \
        python $BPEROOT/apply_bpe.py -c $BPE_CODE > $PATH_TO_DATA_BPE.$l2$i
done;
```

* **Binarization**

We simultaneously binarize the train and test sets.

``` bash
PATH_TO_DICT=...
PATH_TO_DATA_BIN_FOLDER=...
l1=...
l2=...
PATH_TO_TRAIN_DATA_BPE=...
PATH_TO_TEST_DATA_BPE=...

# source + target
fairseq-preprocess \
    --source-lang $l1 --target-lang $l2 \
    --srcdict $PATH_TO_DICT \
    --joined-dictionary \
    --destdir $PATH_TO_DATA_BIN_FOLDER \
    --workers 40 \
    --trainpref $PATH_TO_TRAIN_DATA_BPE \
    --testpref $PATH_TO_TEST_DATA_BPE
# examples
for i in $(seq 1 $N)
do 
 fairseq-preprocess \
     --source-lang $l2$i --target-lang $l1$i \
     --only-source \
     --srcdict $PATH_TO_DICT \
     --joined-dictionary \
     --destdir $PATH_TO_DATA_BIN_FOLDER \
     --workers 40 \
     --trainpref $PATH_TO_TRAIN_DATA_BPE \
     --testpref $PATH_TO_TEST_DATA_BPE
done;
# examples renaming
for split in train test
do
  for i in $(seq 1 $N)
  do
      for suffix in idx bin
      do
          mv $PATH_TO_DATA_BIN_FOLDER/$split.$l2$i-$l1$i.$l2$i.$suffix \
              $PATH_TO_DATA_BIN_FOLDER/$split.$l1-$l2.$l2$i.$suffix
      done;
  done;
done;
```

## Training

Before training, one should execute ```python setup.py build_ext --inplace``` to compile the C++ libraries used during training. Be careful, specific versions of various libraries are required! See [this section](#requirements).

Training should be done using GPUs. For stable training, one should consider using (MAX_TOKENS x NUM_GPU) > 18,000. MAX_TOKEN must be as high as possible considering the GPU memory available.
``` bash
N=...
SAVE_PATH=...
MODEL_NAME=...
PATH_TO_DATA_BIN_FOLDER=...
MAX_ITER=...
MAX_EPOCH=...
MAX_TOKENS=... # batch size

mkdir -p $SAVE_PATH/
mkdir -p $SAVE_PATH/$MODEL

# Train on 8 devices
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUDA_LAUNCH_BLOCKING=1 \
fairseq-train $PATH_TO_DATA_BIN_FOLDER \
  --arch multi_levenshtein_transformer \
  --task multi_translation_lev \
  --num-retrieved $N \
  --max-acceptable-retrieved-ratio 100 \
  --criterion nat_loss \
  --ddp-backend=legacy_ddp \
  --share-all-embeddings \
  --apply-bert-init \
  --decoder-learned-pos \
  --encoder-learned-pos \
  --optimizer adam \
  --adam-betas '(0.9, 0.98)' \
  --clip-norm 0.0 \
  --lr 0.0005 \
  --lr-scheduler inverse_sqrt \
  --stop-min-lr '1e-9' \
  --warmup-init-lr '1e-07' \
  --warmup-updates 10000 \
  --label-smoothing 0.1 \
  --dropout 0.3 \
  --max-tokens $MAX_TOKENS \
  --save-dir $SAVE_PATH/$MODEL_NAME \
  --tensorboard-logdir $SAVE_PATH/$MODEL_NAME \
  --log-format 'simple' \
  --log-interval 200 \
  --save-interval-updates 3000 \
  --keep-interval-updates 10 \
  --no-epoch-checkpoints \
  --max-update $MAX_ITER \
  --max-epoch $MAX_EPOCH \
  --num-workers 2 \
  --source-lang $l1 \
  --target-lang $l2 \
  --fp16 \
  --upsample-primary 1 \
  --max-valency 10 \
  --curriculum-post-del-extra 1 \
  --disable-validation \
  --random-del-prob 0.2 \
  --selection-noise-prob 0.2 \
  --completion-noise-prob 0.2 \
  --nothing-todo-plh-prob 0.4
```

## Inference

Inference should be done on test sets entirely preprocessed as described above.
The results will be stored in INFER_PATH.

* ```--formatted-file```, when not empty, will generate a formatted HTML file to visualize the edition steps on every test sample.
* ```--retain-origin```, when added, will add to the output file information about the origin of the unigram and bi-gram tokens and their precision. Useful to assess the copy rates. It follows the patterns (gen=generated, prec=precision, num=number of):
  * ```PREC-id   copy-prec   gen-prec   num-copy   num-gen   num-total```
  * ```PREC2-id   copy-copy-prec   copy-gen-prec   gen-copy-prec   gen-gen-prec   num-copy-copy   num-copy-gen   num-gen-copy   num-gen-gen```
* ```--realigner``` can be set to several options:
  * ```no```: no realignment
  * ```grad_descent```: realignemnt with gradient descent using regression with a normal distribution to relax the log-likelihood
  * ```grad_descent_multinomial```: **experimental only**, realignemnt with gradient descent using a gaussian mixture distribution to relax the log-likelihood

``` bash
N=...
l1=...
l2=...
MODEL_PATH=...
SCRIPTS=.../mosesdecoder/scripts
DETOKENIZER=$SCRIPTS/tokenizer/detokenizer.perl
INFER_PATH=...
FORMATTED_OUTPUT=...
PATH_TO_DATA_BIN_FOLDER=...

CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 fairseq-generate $PATH_TO_DATA_BIN_FOLDER \
    --task multi_translation_lev \
    --gen-subset test \
    --path $MODEL_PATH \
    --beam 1 \
    --iter-decode-eos-penalty 3. \
    --iter-decode-max-iter 10 \
    --print-step \
    --retain-iter-history \
    --formatted-file $FORMATTED_OUTPUT \
    --batch-size 200 \
    --num-retrieved $N \
    --criterion nat_loss \
    --ddp-backend=legacy_ddp \
    --num-workers 10 \
    --source-lang $l1 \
    --target-lang $l2 \
    --fp16 \
    --max-acceptable-retrieved-ratio 10000 \
    --remove-bpe \
    --upsample-primary 1 \
    --retain-origin \
    --realigner no \
    > $INFER_PATH.$l2

grep ^H $INFER_PATH.$l2 | LC_ALL=C sort -V | cut -f3- | perl $DETOKENIZER -l $l2 -q \
    > $INFER_PATH.hyp.test.$l2

grep ^T $INFER_PATH.$l2 | LC_ALL=C sort -V | cut -f2- | perl $DETOKENIZER -l $l2 -q \
    > $INFER_PATH.ref.test.$l2

grep "^S-" $INFER_PATH.$l2 | LC_ALL=C sort -V | cut -f2- | perl $DETOKENIZER -l $l1 -q \
    > $INFER_PATH.src.test.$l1

grep "^PREC-" $INFER_PATH.$l2 | LC_ALL=C sort -V  \
    > $INFER_PATH.prec.test.$l2

grep "^PREC2-" $INFER_PATH.$l2 | LC_ALL=C sort -V \
    > $INFER_PATH.bi-prec.test.$l2

```

## Requirements

We advice to use a virutal environnement such as conda with the following package versions:

```
python                    3.8

cudatoolkit               11.3.1             
cudatoolkit-dev           11.3.1               
gcc_linux-64              7.3.0              
gxx_linux-64              7.3.0              
libgfortran-ng            12.2.0               
ncurses                   6.3                
ninja                     1.11.0               
numpy                     1.22.3             
omegaconf                 2.0.6                
pytorch                   1.11..8_cuda11.3_cudnn8.2.0_0    
samtools                  1.6                  
scipy                     1.7.3              
tensorboard               2.6.0              
tensorboardx              2.5                  
```

#### conda new environnement
```
conda create -n fairseq python=3.8
conda activate fairseq
```

#### pytorch + CUDA install
```
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```

#### fairseq installation
```
git clone https://github.com/Maxwell1447/fairseq
cd fairseq
pip install -e .
```

#### Various packages
```
conda install \
   numpy=1.22.3 \
   ninja=1.11.0 \
   libgfortran-ng=12.2.0 \
   omegaconf=2.0.6 \
   tensorboard=2.6.0 \
   tensorboardx=2.5 \
   -c conda-forge
conda install \
   gcc_linux-64=7.3.0 \
   gxx_linux-64=7.3.0 \
   ncurses=6.3 \
   scipy=1.7.3 \
   -c anaconda
conda install -c bioconda samtools=1.6
```

#### Compiling C++ libraries
```
python setup.py build_ext --inplace
```

# Original code is from https://github.com/pytorch/fairseq/blob/master/examples/roberta/README.glue.md

TOTAL_NUM_UPDATES=2036  # original 2036 - 10 epochs through RTE for bsz 16
WARMUP_UPDATES=122      # 122 6 percent of the number of updates
LR=1e-06                # 1e-06 is original Peak LR for polynomial LR scheduler.
NUM_CLASSES=3
MAX_SENTENCES=12        # used to be 12 Batch size.
UPDATE_FREQ=8          # 8 increase the batchsize
#ROBERTA_PATH=/opt/models/checkpoints2/

CUDA_VISIBLE_DEVICES=0

## --restore-file $ROBERTA_PATH \
## fairseq-train MNLI-bin/ \

fairseq-train MNLI-bin/ \
    --max-positions 512 \
    --max-sentences $MAX_SENTENCES \
    --update-freq $UPDATE_FREQ \
    --max-tokens 4400 \
    --task sentence_prediction \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --init-token 0 --separator-token 2 \
    --arch roberta_base \
    --criterion sentence_prediction \
    --num-classes $NUM_CLASSES \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-6 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
    --patience 2 \
    --no-epoch-checkpoints \
    --find-unused-parameters \
    --restore-file /opt/models/checkpoints2/wikitext-pre-ro-base.pt \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --num-workers 16;

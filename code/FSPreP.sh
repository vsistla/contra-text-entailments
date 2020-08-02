fairseq-preprocess \
    --only-source \
    --srcdict gpt2_bpe/dict.txt \
    --trainpref mnli/$1.train.bpe \
    --validpref mnli/$1.valid.bpe \
    --testpref mnli/$1.test.bpe \
    --destdir data-bin/$1 \
    --workers 60


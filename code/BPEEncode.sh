for SPLIT in train valid test; do \
  python -m examples.roberta.multiprocessing_bpe_encoder \
        --encoder-json gpt2_bpe/encoder.json \
        --vocab-bpe gpt2_bpe/vocab.bpe \
        --inputs mnli/$1.${SPLIT}.txt \
        --outputs mnli/$1.${SPLIT}.bpe \
        --keep-empty \
        --workers 60
done


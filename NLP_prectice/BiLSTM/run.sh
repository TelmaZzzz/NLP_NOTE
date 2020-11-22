python main.py \
--train-data="CoNLL2003_NER/train" \
--valid-data="CoNLL2003_NER/valid" \
--test-data="CoNLL2003_NER/test" \
--embed-dim=133 \
> train.log 2>&1 &

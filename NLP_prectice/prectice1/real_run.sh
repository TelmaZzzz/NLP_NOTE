TRAIN_LOAD_PATH='/home/telma/work/NLP_NOTE/NLP_prectice/prectice1/aclImdb/train/train.csv'
TEST_LOAD_PATH='/home/telma/work/NLP_NOTE/NLP_prectice/prectice1/aclImdb/test/test.csv'
SAVE_PATH='/home/telma/work/NLP_NOTE/NLP_prectice/prectice1/model/'
python main.py --train-load-path=${TRAIN_LOAD_PATH} \
               --test-load-path=${TEST_LOAD_PATH} \
               --save-path=${SAVE_PATH}
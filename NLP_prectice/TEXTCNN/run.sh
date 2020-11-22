ROLE="train"
LOAD_PATH="/home/telma/work/NLP_NOTE/NLP_prectice/prectice1/aclImdb/${ROLE}"
FILE_NAME="${ROLE}"
python gen_data.py --load-path=${LOAD_PATH} \
                   --file-name=${FILE_NAME} \
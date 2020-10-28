ROLE="train"
LOAD_PATH="/Users/bytedance/work/NLP_NOTE/NLP_prectice/aclImdb/${ROLE}"
FILE_NAME="${ROLE}"
python gen_data.py --load-path=${LOAD_PATH} \
                   --file-name=${FILE_NAME} \
#!/bin/bash

# dataset consts
DATASET_URL="https://www.kaggle.com/api/v1/datasets/download/tm21cy/nyt-connections"
DATASET_FILE_NAME="ConnectionsFinalDataset (1).json"

# download dataset
mkdir ./tmp
curl -L -o ./tmp/game_data.zip $DATASET_URL
unzip ./tmp/game_data.zip -d ./tmp

# json transforms
jq '[.[] | { words: [.words[] | ascii_downcase], answer_key: [.answers[] | [.words[] | ascii_downcase]] }]' "./tmp/$DATASET_FILE_NAME" > ./data/game_data.json
jq "[.[] | .words] | flatten | unique" ./data/game_data.json > ./data/all_words.json

# cleanup
rm -rf ./tmp


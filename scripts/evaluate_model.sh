#!/bin/bash

set -e # fail on command failures

declare -a SCORERS=(
    # "mean_centroid_distance"
    # "mean_projective_centroid_distance"
    # "max_centroid_distance"
    # "max_projective_centroid_distance"
    "max_pairwise_cosine_distance"
    "mean_pairwise_cosine_distance"
    "max_pairwise_l2_distance"
    "mean_pairwise_l2_distance"
    "cosine_silhouette_coefficient"
    "l2_silhouette_coefficient"
    "lopez_mcdonald_emami"
)

if [ -z "$1" ]; then
    echo "Provide a model name. Options can be found here: https://sbert.net/docs/sentence_transformer/pretrained_models.html"
    exit 1
fi

# load embedding for model
if [ ! -f ./data/$(echo $1 | sed -e "s/\//-/g").json ]; then
    echo "Loading embeddings for Model[$1]..."
    python3 ./src/load_embeddings.py --model "$1"
    echo "Embeddings loaded!"
    echo "=========================================================="
fi

# run the model for each scorer
for scorer in "${SCORERS[@]}"
do
    echo "Evaluating Model[$1] for Scorer[$scorer]"
    python3 ./src/run_game.py --scorer "$scorer" --model "$1"
    echo "Model[$1] Scorer[$scorer]. Average number of mistakes: $(jq '[.game_results[] | .turns_taken] | add / length' "./data/results/$(echo $1 | sed -e "s/\//-/g")_$scorer.json")"
    echo "=========================================================="
done
wait


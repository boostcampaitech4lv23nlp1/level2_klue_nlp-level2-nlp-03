#!/bin/bash
CONFIGS=("Query_entity_marker_punct" "Query_entity_marker" "Query_entity_mask" "Query_typed_entity_marker_punct" "Query_typed_entity_marker")

for (( i=0; i<5; i++ ))
do
    python3 train.py --config ${CONFIGS[$i]}
done
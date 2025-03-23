#!/bin/bash

declare -a videos=(
                   "basketball.mp4"
                   "sing.mp4"
                   )

declare -a prompts=(
                    "a man playing basketball in a city with buildings and flagpoles behind"
                    "a man singing on stage with others"
                    )

arraylength=${#videos[@]}

for (( i=0; i<${arraylength}; i++ ));
do
  python3 ./generate_data.py --file "${videos[$i]}" --prompt "${prompts[$i]}"
done
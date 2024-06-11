#!/bin/bash

# Generate the list of sizes
sizes=($(seq 100 300 10000))

# Iterate over the sizes
for ((i=0; i<${#sizes[@]}; i++)); do
    # Generate the name based on the size
    name="square${sizes[i]}"
    # Run the Python script with arguments

    python ddpm.py --dataset square --experiment_name "$name" --num_epochs 100 --dataset_size "${sizes[i]}" --beta_schedule ours --num_timesteps 1000
done


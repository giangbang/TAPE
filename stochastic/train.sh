#!/bin/sh
algo="offpg_smac"
seed_max=2

scenarios=( "lbforaging:Foraging-15x15-4p-3f-v3" "lbforaging:Foraging-15x15-3p-4f-v3" "lbforaging:Foraging-10x10-3p-3f-v3" )
total_times=( 5_000_000 )
# scenarios=( "lbforaging:Foraging-15x15-3p-4f-v3" )

n_exp=${#scenarios[@]}

for ((i=0; i<n_exp; i++)); do
    for seed in `seq ${seed_max}`;
    do
        scenario="${scenarios[$i]}"
        total_ts=5_000_000
        python src/main.py --config=${algo} --env-config=gymma with env_args.time_limit=50 \
                env_args.key="${scenario}" t_max=${total_ts}   p=0.3
    done
done
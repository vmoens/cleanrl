#!/bin/bash
# Execute scripts with different seeds and additional arguments for torchcompile scripts
scripts=(
#    dqn.py
#    dqn_jax.py
#    dqn_torchcompile.py
#    ppo_atari.py
    ppo_atari_envpool.py
    ppo_atari_envpool_torchcompile.py
    ppo_atari_envpool_xla_jax.py
    sac_continuous_action.py
    sac_continuous_action_torchcompile.py
    td3_continuous_action.py
    td3_continuous_action_jax.py
    td3_continuous_action_torchcompile.py
)
for seed in 1 2 3; do
    for script in "${scripts[@]}"; do
        if [[ $script == *_torchcompile.py ]]; then
            python $script --seed=$seed --cudagraphs --compile
        else
            python $script --seed=$seed
        fi
    done
done

if [ $# -lt 3 ]:
then
    echo "./$0 tag env_name alg_name [gpu_id=0]"
    echo "example: ./$0 test spread AORPO [0]"
    exit
fi

if [ $# -lt 4 ]:
then
    gpu=0
else
    gpu=${4}
fi

if [ ${2} = "spread" ]:
then
    set -x
    for seed in 0 1 2 3 4
    do 
        CUDA_VISIBLE_DEVICES=${gpu} python main.py simple_spread ${1} --cuda --gpu_rollout_model --algorithm ${3} --model_hidden_dim 512 --model_lr 0.001 --n_epoch 100 --n_model_warmup_episode 1500 --opp_lr 0.0005  --dynamics_model_update_step_interval 3750 --Env_rate_n_epoch 40 --Env_rate_start 0.5 --Env_rate_finish 0.5 --G 20 --M 1024 --model_lr_schedule_steps 8000 --K 15 --A 5 --B 25 --seed ${seed}
    done
elif [ ${2} = "speaker_listener" ]:
then
    set -x
    for seed in 0 1 2 3 4
    do 
        CUDA_VISIBLE_DEVICES=${gpu} python main.py simple_speaker_listener ${1} --cuda --gpu_rollout_model --algorithm ${3} --model_hidden_dim 512 --model_lr 0.001 --n_epoch 40 --n_model_warmup_episode 1500 --opp_lr 0.0005 --dynamics_model_update_step_interval 3750 --Env_rate_n_epoch 40 --Env_rate_start 0.5 --Env_rate_finish 0.5 --G 10 --M 1024 --K 10 --A 3 --B 25 --lr 0.005 --model_lr_schedule_steps 4000 --seed ${seed}
    done
elif [ ${2} = "schedule" ]:
then
    set -x
    for seed in 0 1 2 3 4
    do 
        CUDA_VISIBLE_DEVICES=${gpu} python main.py simple_schedule ${1} --cuda --gpu_rollout_model --algorithm ${3} --model_hidden_dim 512 --model_lr 0.001 --n_epoch 50 --n_model_warmup_episode 1500 --opp_lr 0.0005  --dynamics_model_update_step_interval 3750 --Env_rate_n_epoch 40 --Env_rate_start 0.5 --Env_rate_finish 0.5 --G 20 --M 1024 --K 10 --A 5 --B 25 --seed ${seed}
    done
fi
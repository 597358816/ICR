set -x

export VERL_HEAD_ONLY_TRAINING=1
MODEL_PATH=/home/dataset-assist-0/wc/models/Qwen/Qwen2.5-3B-Instruct  # replace it with your local file path

FORMAT_PROMPT="""You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
 The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}."""
NAME="qwen2.5-3b-Ins-overall"
python3 -m verl.trainer.main \
    config=config.yaml \
    data.train_files=hiyouga/math12k@train \
    data.val_files=hiyouga/math12k@test \
    data.format_prompt="${FORMAT_PROMPT}" \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.project_name=LZC \
    trainer.experiment_name="${NAME}" \
    trainer.n_gpus_per_node=2 \
    data.max_response_length=2048 \
    worker.rollout.n=5 \
    worker.actor.micro_batch_size_per_device_for_update=4 \
    worker.actor.micro_batch_size_per_device_for_experience=8 \
    trainer.save_limit=4 \
    trainer.save_freq=20 \
    trainer.total_episodes=2 \
    worker.actor.global_batch_size=64 \
    data.rollout_batch_size=128 \
    trainer.val_freq=-1 \
    trainer.save_checkpoint_path="/home/dataset-assist-0/wc/checkpoints/Qwen2.5-3B/${NAME}" \


    # worker.actor.micro_batch_size_per_device_for_update=16 \
    # worker.actor.micro_batch_size_per_device_for_experience=32 \
    #trainer.save_freq=-1
    #data.train_files=/home/dataset-assist-0/wc/data/dapo/train_dapo.parquet \
    #trainer.load_checkpoint_path=/home/dataset-assist-0/wc/EasyR1-main/examples/checkpoints/Entropy-Controller/IS-e0.5-a0.1-ratio-noclipdual/global_step_110 \


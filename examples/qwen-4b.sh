set -x
export WANDB_BASE_URL=https://api.bandw.top
export WANDB_MODE=online
export WANDB_API_KEY="b80b9192efe12f9fc47ef0fc711bde76686fb981"
export TMPDIR=/vepfs-mlp2/c20250203/250602012/tmp
export PIP_CACHE_DIR=/vepfs-mlp2/c20250203/250602012/cache
MODEL_PATH=/vepfs-mlp2/c20250203/250602012/models/Qwen/Qwen3-4B  # replace it with your local file path
NAME="qwen3-4b-ICR-ab-neg2"

FORMAT_PROMPT="""You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}."""
#FORMAT_PROMPT="""Solve the problem and provide only the final answer. Do NOT include any reasoning steps, explanations, or intermediate calculations. The final answer MUST be enclosed in \boxed{}."""

/vepfs-mlp2/c20250203/250602012/Anaconda/envs/easyr1/bin/python -m verl.trainer.main \
    config=/vepfs-mlp2/c20250203/250602012/EasyR1/examples/config.yaml \
    worker.actor.model.model_path=${MODEL_PATH} \
    data.train_files=/vepfs-mlp2/c20250203/250602012/data/train_dapo.parquet \
    data.val_files=/vepfs-mlp2/c20250203/250602012/data/math/data/test-00000-of-00001.parquet \
    data.max_response_length=8192 \
    data.rollout_batch_size=128 \
    data.format_prompt="${FORMAT_PROMPT}" \
    worker.rollout.n=8 \
    worker.rollout.max_num_batched_tokens=10240 \
    trainer.experiment_name="${NAME}" \
    trainer.project_name="Length" \
    trainer.val_freq=-1 \
    trainer.save_limit=1 \
    trainer.save_freq=10 \
    trainer.total_episodes=2 \
    trainer.val_before_train=false \
    worker.actor.micro_batch_size_per_device_for_update=2 \
    worker.actor.micro_batch_size_per_device_for_experience=4 \
    worker.actor.global_batch_size=64 \
    trainer.save_checkpoint_path="/vepfs-mlp2/c20250203/250602012/checkpoints/Qwen3-4B/${NAME}" \
    worker.reward.length_reward="LP1" \
    trainer.load_checkpoint_path=/vepfs-mlp2/c20250203/250602012/checkpoints/Qwen3-4B/${NAME}/global_step_120 \
    # worker.reward.length_reward="LP1" \
    # trainer.n_gpus_per_node=4
    # trainer.load_checkpoint_path=/home/dataset-assist-0/wc/checkpoints/Qwen3-4B/qwen3-4b-LP1/global_step_130 \
    
    # worker.actor.use_entropy_loss=true \
    # data.rollout_batch_size=64 \
    # worker.actor.global_batch_size=32 \
    # worker.actor.micro_batch_size_per_device_for_update=1 \
    # worker.actor.micro_batch_size_per_device_for_experience=2 \
    # trainer.total_episodes=2 \
    # trainer.save_freq=20 \

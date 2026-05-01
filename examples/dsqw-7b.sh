set -x
MODEL_PATH=deepseek/DeepSeek-R1-Distill-Qwen-7B  # replace it with your local file path
NAME="dsqw-7b-ICR"
FORMAT_PROMPT="""You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
 The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}."""
python3 -m verl.trainer.main \
    config=config.yaml \
    data.train_files=BytedTsinghua-SIA/DAPO-Math-17k \
    data.val_files=hiyouga/math12k@test \
    data.format_prompt="${FORMAT_PROMPT}" \
    data.max_response_length=8192 \
    worker.rollout.max_num_batched_tokens=10240 \
    worker.rollout.n=8 \
    worker.actor.model.model_path=${MODEL_PATH} \
    data.rollout_batch_size=192 \
    worker.actor.global_batch_size=96 \
    worker.actor.micro_batch_size_per_device_for_update=4 \
    worker.actor.micro_batch_size_per_device_for_experience=8 \
    trainer.experiment_name="${NAME}" \
    trainer.n_gpus_per_node=8 \
    trainer.save_limit=8 \
    trainer.save_freq=20 \
    trainer.total_episodes=2 \
    trainer.val_before_train=false \
    trainer.algorithm="ICR" \
    worker.reward.length_reward="LP-F" \

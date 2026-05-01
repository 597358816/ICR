set -x
MODEL_PATH=Qwen/Qwen3-4B  # replace it with your local file path
NAME="qwen3-4b-ICR"

FORMAT_PROMPT="""You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}."""
#FORMAT_PROMPT="""Solve the problem and provide only the final answer. Do NOT include any reasoning steps, explanations, or intermediate calculations. The final answer MUST be enclosed in \boxed{}."""

python3 -m verl.trainer.main \
    config=config.yaml \
    worker.actor.model.model_path=${MODEL_PATH} \
    data.train_files=BytedTsinghua-SIA/DAPO-Math-17k \
    data.val_files==hiyouga/math12k@test \
    data.max_response_length=8192 \
    data.rollout_batch_size=128 \
    data.format_prompt="${FORMAT_PROMPT}" \
    worker.rollout.n=8 \
    worker.rollout.max_num_batched_tokens=10240 \
    trainer.experiment_name="${NAME}" \
    trainer.project_name="Length" \
    trainer.val_freq=-1 \
    trainer.save_limit=8 \
    trainer.save_freq=20 \
    trainer.total_episodes=2 \
    trainer.val_before_train=false \
    worker.actor.micro_batch_size_per_device_for_update=2 \
    worker.actor.micro_batch_size_per_device_for_experience=4 \
    worker.actor.global_batch_size=64 \
    trainer.algorithm="ICR" \
    worker.reward.length_reward="LP-F" \

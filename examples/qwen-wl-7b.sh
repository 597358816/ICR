set -x

MODEL_PATH=/home/dataset-assist-0/wc/models/Qwen/Qwen2.5-VL-7B-Instruct  # replace it with your local file path

FORMAT_PROMPT="""You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
 The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}."""

python3 -m verl.trainer.main \
    config=config.yaml \
    data.train_files=WaltonFuture/MMMT-ThinkLite-3k-random@train \
    data.val_files=hiyouga/geometry3k@test \
    data.max_pixels=1000000 \
    data.format_prompt="${FORMAT_PROMPT}" \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.project_name=ReplayEntropy \
    trainer.experiment_name=KL-cov \
    worker.actor.micro_batch_size_per_device_for_update=2 \
    worker.actor.micro_batch_size_per_device_for_experience=4 \
    trainer.save_freq=10 \
    trainer.save_freq=12 \
    trainer.total_episodes=20 \
    trainer.n_gpus_per_node=8 \
    worker.actor.use_entropy_loss=true \
    worker.actor.entropy_coef=0.01

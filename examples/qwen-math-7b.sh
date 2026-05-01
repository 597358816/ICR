set -x

MODEL_PATH=/home/dataset-assist-0/wc/models/Qwen/Qwen2.5-math-7B  # replace it with your local file path
NAME="qwen-math-7b-ab-negative" 
FORMAT_PROMPT="""You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
 The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}."""

python3 -m verl.trainer.main \
    config=config.yaml \
    data.train_files=/home/dataset-assist-0/wc/data/dapo/train_dapo.parquet \
    data.val_files=hiyouga/math12k@test \
    data.format_prompt="${FORMAT_PROMPT}" \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.micro_batch_size_per_device_for_update=16 \
    worker.actor.micro_batch_size_per_device_for_experience=32 \
    trainer.experiment_name="${NAME}" \
    trainer.project_name="new-AEPO" \
    trainer.n_gpus_per_node=8 \
    trainer.save_limit=8 \
    trainer.save_freq=20 \
    trainer.total_episodes=10 \
    trainer.val_before_train=false \
    trainer.save_checkpoint_path="/home/dataset-assist-0/wc/checkpoints/Qwen2.5-math-7B/${NAME}" \
    #worker.actor.use_entropy_loss=true \
    #worker.actor.entropy_coef=0.03 \
    #trainer.save_freq=-1
    #trainer.load_checkpoint_path=/home/dataset-assist-0/wc/EasyR1-main/examples/checkpoints//home/dataset-assist-0/wc/EasyR1-main/examples/checkpoints/easyr1/math-ab-reinforce/global_step_160 \
    #trainer.save_freq=-1 

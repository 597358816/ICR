set -x

MODEL_PATH=/home/dataset-assist-0/wc/models/Qwen/Qwen2.5-7B  # replace it with your local file path
NAME="qwen2.5-7b-Entreg0.1"
FORMAT_PROMPT="""You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
 The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}."""

python3 -m verl.trainer.main \
    config=config.yaml \
    data.train_files=/home/dataset-assist-0/wc/data/dapo/train_dapo.parquet \
    data.val_files=hiyouga/math12k@test \
    data.format_prompt="${FORMAT_PROMPT}" \
    data.max_response_length=2048 \
    worker.rollout.n=8 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.micro_batch_size_per_device_for_update=8 \
    worker.actor.micro_batch_size_per_device_for_experience=16 \
    trainer.experiment_name="${NAME}" \
    trainer.project_name="new-AEPO" \
    trainer.n_gpus_per_node=8 \
    trainer.save_limit=8 \
    trainer.save_freq=20 \
    trainer.total_episodes=6 \
    trainer.val_before_train=false \
    trainer.val_freq=-1 \
    trainer.save_checkpoint_path="/home/dataset-assist-0/wc/checkpoints/Qwen2.5-7B/${NAME}" \
    worker.actor.use_entropy_loss=true \
    worker.actor.entropy_coef=0.01
    #worker.rollout.max_num_batched_tokens=10240 \
    #worker.actor.use_entropy_loss=true 
    #algorithm.disable_kl=false \
    
    #trainer.save_freq=-1
    #trainer.load_checkpoint_path=/home/dataset-assist-0/wc/EasyR1-main/examples/checkpoints/Entropy-Controller/IS-e0.5-a0.1-ratio-noclipdual/global_step_110 \
    #trainer.save_freq=-1 

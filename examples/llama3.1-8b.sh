set -x

MODEL_PATH=/home/dataset-assist-0/wc/models/meta-llama/Llama-3.1-8B-Instruct  # replace it with your local file path
NAME="llama3.1-8b"
FORMAT_PROMPT="""You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
 The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}."""

python3 -m verl.trainer.main \
    config=config.yaml \
    data.train_files=hiyouga/math12k@train \
    data.val_files=hiyouga/math12k@test \
    data.max_response_length=2048 \
    data.format_prompt="${FORMAT_PROMPT}" \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.micro_batch_size_per_device_for_update=8 \
    worker.actor.micro_batch_size_per_device_for_experience=16 \
    trainer.project_name=EFRame@k \
    trainer.save_freq=10 \
    trainer.save_limit=8 \
    trainer.val_freq=-1 \
    trainer.experiment_name="${NAME}" \
    trainer.save_checkpoint_path="/home/dataset-assist-0/wc/checkpoints/Llama-3.1-8B-Instruct/${NAME}" \
    trainer.total_episodes=4 \
    trainer.n_gpus_per_node=8 \
    trainer.val_before_train=false \
    #algorithm.disable_kl=false
    #worker.actor.use_entropy_loss=true \
    #worker.actor.entropy_coef=0.01
    #trainer.load_checkpoint_path="/home/dataset-assist-0/wc/checkpoints/Llama-3.1-8B-Instruct/${NAME}"/global_step_60 \


    #trainer.save_freq=-1
    #trainer.load_checkpoint_path=/home/dataset-assist-0/wc/EasyR1-main/examples/checkpoints/Entropy-Controller/IS-e0.5-a0.1-ratio-noclipdual/global_step_110 \
    #trainer.save_freq=-1 

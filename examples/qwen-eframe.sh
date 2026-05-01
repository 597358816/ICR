set -x

MODEL_PATH=/home/dataset-assist-0/wc/models/Qwen/Qwen2.5-math-7B  # replace it with your local file path

FORMAT_PROMPT="""You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
 The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}."""

python3 -m verl.trainer.main \
    config=config.yaml \
    data.train_files=/home/dataset-assist-0/wc/data/dapo/train_dapo.parquet \
    data.val_files=hiyouga/math12k@test \
    data.format_prompt="${FORMAT_PROMPT}" \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.project_name=EFRame@k \
    trainer.experiment_name=DAPO \
    trainer.n_gpus_per_node=8 \
    trainer.total_episodes=6 \
    trainer.save_limit=10 \



    #trainer.save_freq=-1
    #trainer.load_checkpoint_path=/home/dataset-assist-0/wc/EasyR1-main/examples/checkpoints/Entropy-Controller/IS-e0.5-a0.1-ratio-noclipdual/global_step_110 \
    #trainer.save_freq=-1 

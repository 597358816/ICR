# for step in $(seq 200 40 340)
# do
#     echo ${step}
#     python3 /home/dataset-assist-0/wc/EasyR1/scripts/model_merger.py --local_dir /home/dataset-assist-0/wc/checkpoints/Qwen2.5-3B/qwen2.5-3b-Ins-head/global_step_${step}/actor/
# done

python3 /home/dataset-assist-0/wc/EasyR1/scripts/model_merger.py --local_dir /home/dataset-assist-0/wc/checkpoints/Qwen2.5-7B/qwen2.5-7b-CISPO/global_step_20/actor/
python3 /home/dataset-assist-0/wc/EasyR1/scripts/model_merger.py --local_dir /home/dataset-assist-0/wc/checkpoints/Qwen2.5-7B/qwen2.5-7b-CISPO/global_step_40/actor/
python3 /home/dataset-assist-0/wc/EasyR1/scripts/model_merger.py --local_dir /home/dataset-assist-0/wc/checkpoints/Qwen2.5-7B/qwen2.5-7b-CISPO/global_step_60/actor/
python3 /home/dataset-assist-0/wc/EasyR1/scripts/model_merger.py --local_dir /home/dataset-assist-0/wc/checkpoints/Qwen2.5-7B/qwen2.5-7b-CISPO/global_step_80/actor/



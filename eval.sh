module load cuda/11.7
echo "Loaded cuda cudnn"
source /home/pmayilvahanan/.llm_line/bin/activate
echo "activated source from /home/pmayilvahanan/.llm_line/bin/activate"
export HF_HOME=/tmp/  # tmp is faster and works better
echo "Set HF_HOME to $HF_HOME"
python3 main.py --models_yaml configs/one_model.yaml --shuffle --compute-both
echo DONE.

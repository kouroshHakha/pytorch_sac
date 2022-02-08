
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export D4RL_SUPPRESS_IMPORT_ERROR=1
export D4RL_DATASET_DIR=./d4rl_datasets
export PYTHONPATH=./:$PYTHONPATH

source .wandbrc

export OPENAI_API_KEY='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'
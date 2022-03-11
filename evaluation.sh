DIR=coqa_two_gpt_medium_Checkpoint_1

python tool/split_hyp_ref.py $DIR

python tool/evaluate.py $DIR


import os

# for BERT target model
command = 'python attack_nli.py --dataset_path data/snli ' \
          '--target_model bert ' \
          '--target_model_path /scratch/jindi/adversary/BERT/results/SNLI ' \
          '--counter_fitting_embeddings_path /data/medg/misc/jindi/nlp/embeddings/counter-fitted-vectors.txt ' \
          '--counter_fitting_cos_sim_path /scratch/jindi/adversary/cos_sim_counter_fitting.npy ' \
          '--USE_cache_path /scratch/jindi/tf_cache ' \
          '--output_dir results/snli_bert'

command = 'python3 attack_nli.py --dataset_path data/snli ' \
          '--target_model bert ' \
          '--counter_fitting_embeddings_path /datadrive/axon/asistant/TextFooler/counter-fitted-vectors.txt ' \
          '--target_model_path /datadrive/axon/asistant/uncased_L-12_H-768_A-12/ ' \
          '--USE_cache_path /datadrive/axon/asistant/TextFooler/tf_cache ' \
          '--output_dir /datadrive/axon/asistant/TextFooler/results'

os.system(command)
#!/bin/bash

# Entire wikipedia passages set obtain by splitting all pages into 100-word segments (no overlap)
python /Volumes/'T7 Shield'/project/DPR_Finetuning/DPR/dpr/data/download_data.py \
	--resource data.wikipedia_split.psgs_w100 \
	--output_dir /Volumes/'T7 Shield'/project/DPR_Finetuning/DPR

# NQ dev subset with passages pools for the Retriever train time validation
python /Volumes/'T7 Shield'/project/DPR_Finetuning/DPR/dpr/data/download_data.py \
	--resource data.retriever.nq-dev \
	--output_dir /Volumes/'T7 Shield'/project/DPR_Finetuning/DPR

# NQ train subset with passages pools for the Retriever training
python /Volumes/'T7 Shield'/project/DPR_Finetuning/DPR/dpr/data/download_data.py \
	--resource data.retriever.nq-train \
	--output_dir /Volumes/'T7 Shield'/project/DPR_Finetuning/DPR

# NQ dev subset for Retriever validation and IR results generation
python /Volumes/'T7 Shield'/project/DPR_Finetuning/DPR/dpr/data/download_data.py \
	--resource data.retriever.qas.nq-dev \
	--output_dir /Volumes/'T7 Shield'/project/DPR_Finetuning/DPR

# NQ test subset for Retriever validation and IR results generation
python /Volumes/'T7 Shield'/project/DPR_Finetuning/DPR/dpr/data/download_data.py \
	--resource data.retriever.qas.nq-test \
	--output_dir /Volumes/'T7 Shield'/project/DPR_Finetuning/DPR

# NQ train subset for Retriever validation and IR results generation
python /Volumes/T7 Shield/project/DPR_Finetuning/DPR/dpr/data/download_data.py \
	--resource data.retriever.qas.nq-train \
	--output_dir /Volumes/'T7 Shield'/project/DPR_Finetuning/DPR

# Encoded wikipedia files using a biencoder checkpoint(checkpoint.retriever.single.nq.bert-base-encoder) trained on NQ dataset
python /Volumes/'T7 Shield'/project/DPR_Finetuning/DPR/dpr/data/download_data.py \
	--resource data.retriever_results.nq.single.wikipedia_passages \
	--output_dir /Volumes/'T7 Shield'/project/DPR_Finetuning/DPR

# Retrieval results of NQ test dataset for the encoder trained on NQ
python /Volumes/'T7 Shield'/project/DPR_Finetuning/DPR/dpr/data/download_data.py \
	--resource data.retriever_results.nq.single.test \
	--output_dir /Volumes/'T7 Shield'/project/DPR_Finetuning/DPR

# Retrieval results of NQ dev dataset for the encoder trained on NQ
python /Volumes/'T7 Shield'/project/DPR_Finetuning/DPR/dpr/data/download_data.py \
	--resource data.retriever_results.nq.single.dev \
	--output_dir /Volumes/'T7 Shield'/project/DPR_Finetuning/DPR

# Retrieval results of NQ train dataset for the encoder trained on NQ
python /Volumes/'T7 Shield'/project/DPR_Finetuning/DPR/dpr/data/download_data.py \
	--resource data.retriever_results.nq.single.train \
	--output_dir /Volumes/'T7 Shield'/project/DPR_Finetuning/DPR

# Biencoder weights trained on NQ data and HF bert-base-uncased model
python /Volumes/'T7 Shield'/project/DPR_Finetuning/DPR/dpr/data/download_data.py \
	--resource checkpoint.retriever.single.nq.bert-base-encoder \
	--output_dir /Volumes/'T7 Shield'/project/DPR_Finetuning/DPR

# # DPR index on NQ-single retriever
# python /Volumes/'T7 Shield'/project/DPR_Finetuning/DPR/dpr/data/download_data.py \
# 	--resource indexes.single.nq.full.index \
#	--output_dir /Volumes/'T7 Shield'/project/DPR_Finetuning/DPR

# # DPR index on NQ-single retriever (metadata)
# python /Volumes/'T7 Shield'/project/DPR_Finetuning/DPR/dpr/data/download_data.py \
# 	--resource indexes.single.nq.full.index_meta \
#	--output_dir /Volumes/'T7 Shield'/project/DPR_Finetuning/DPR

# # DPR index on NQ-single retriever when only Wikipedia pages seen during training are considered
# python /Volumes/'T7 Shield'/project/DPR_Finetuning/DPR/dpr/data/download_data.py \
# 	--resource indexes.single.nq.subset.index \
#	--output_dir /Volumes/'T7 Shield'/project/DPR_Finetuning/DPR

# # DPR index on NQ-single retriever when only Wikipedia pages seen during training are considered (metadata)
# python /Volumes/'T7 Shield'/project/DPR_Finetuning/DPR/dpr/data/download_data.py \
# 	--resource indexes.single.nq.subset.index_meta \
#	--output_dir /Volumes/'T7 Shield'/project/DPR_Finetuning/DPR
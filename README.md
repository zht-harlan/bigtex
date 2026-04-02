# BiGTex
# Integrating Structural and Semantic Signals in Text-Attributed Graphs with BiGTex


## Install requirements in  `requirements.txt`

## Download TAG datasets



| Dataset | Description |
| ----- |  ---- |
| ogbn-arxiv  | The [OGB](https://ogb.stanford.edu/docs/nodeprop/) provides the mapping from MAG paper IDs into the raw texts of titles and abstracts. <br/>Download the dataset [here](https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz), move it to `datasets/arxiv`.|
| ogbn-products |  Download the dataset [here](https://drive.usercontent.google.com/download?id=1gsabsx8KR2N9jJz16jTcA0QASXsNuKnN&export=download&authuser=0), unzip and move them to `datasets/products`.|
| ogbn-products (subset) |  Download the dataset [here](https://drive.google.com/file/d/1F9D9NauJMlmwGcmLxhwbyAhfguWEZApA/view?usp=drive_link), unzip and move them to `datasets/products`.|
| arxiv_2023 |  Download the dataset [here](https://drive.google.com/file/d/1ekG96JHNPWqeQdb6D_GZoM28OGRLdcS_/view?usp=drive_link), unzip and move it to `datasets/arxiv_2023`.|
| PubMed | Download the dataset [here](https://drive.google.com/file/d/1sYZX-jP6H8OkopVa9cp8-KXdEti5ki_W/view?usp=sharing), unzip and move it to `datasets/pubmed`.|
| Photo | Download the dataset (photo.pt) [here](https://drive.google.com/drive/folders/1bSRCZxt0c11A3717DYDjO112fo_zC8Ec), unzip and move it to `datasets/photo`.|


## Training and then save embeddings for BiGTex and ogbn-arxiv

```
python main.py 'arxiv' 'BiGTex'
```
you can run for other dataset: 'cora', 'pubmed', 'products', 'arxiv_2023'
or other models: 'MLP', 'GCN', 'GAT', 'SAGE'

## Batch benchmark on server

Run the four requested datasets for 5 runs each and export CSV files:

```bash
bash run_server_benchmarks.sh <your_conda_env_name>
```

This writes per-dataset CSV files under `benchmark_results/<dataset>/` and a merged summary to
`benchmark_results/benchmark_summary.csv`.

## Stage-3 purified graph encoder

Train a graph encoder from cached offline text embeddings:

```bash
python train_purified_graph_encoder.py ogbn-arxiv \
  --artifact_root offline_artifacts \
  --hidden_dim 256 \
  --gnn_type sage \
  --num_layers 2 \
  --epochs 30 \
  --save_ze
```

This reads `offline_artifacts/<dataset>/text_embeddings.pt`, trains `embedding -> MLP -> GNN -> Ze`,
and writes run/summary CSV files plus the best checkpoint under `purified_graph_results/<dataset>/`.

## Stage-4 residual quantized graph encoder

Train a quantized graph encoder from cached offline text embeddings:

```bash
python train_quantized_purified_graph_encoder.py ogbn-arxiv \
  --artifact_root offline_artifacts \
  --hidden_dim 256 \
  --gnn_type sage \
  --num_layers 2 \
  --num_quantizers 3 \
  --codebook_size 128 \
  --epochs 30 \
  --save_quantized_artifacts
```

This trains `embedding -> MLP -> GNN -> Ze -> residual quantizer -> Zq -> classifier`,
and writes CSV files, the best checkpoint, plus optional `Ze / Zq / code indices` under
`quantized_graph_results/<dataset>/`.

## Stage-5 cross-modal fusion classifier

Train the final model with structure tokens and refined text tokens:

```bash
python train_quantized_graph_text_classifier.py ogbn-arxiv \
  --artifact_root offline_artifacts \
  --backbone_name scibert \
  --hidden_dim 256 \
  --gnn_type sage \
  --num_layers 2 \
  --num_quantizers 3 \
  --codebook_size 128 \
  --epochs 10
```

This trains `embedding -> MLP -> GNN -> Ze -> residual quantizer -> struct tokens + text tokens -> PLM`,
and writes CSV files plus the best checkpoint under `fusion_results/<dataset>/`.

## BiGTex embeddings
You can download the generated embeddings by BiGTex [here ](https://drive.google.com/file/d/1RKJEHeN_lhO7drEd4KlofAqiTqmzSWEE/view?usp=drive_link).
unzip and move them to `embeddings`, so you can run more experiments like link prediction or clusstering using them.


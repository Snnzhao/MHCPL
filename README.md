# MHCPL

The implementation of _Multi-view Hypergraph Contrastive Policy Learning for  Conversational Recommendation_ (SIGIR 2023). 

More descriptions are available via the [paper](https://dl.acm.org/doi/abs/10.1145/3539618.3591737).

The code is partially referred to [MCMIPL](https://github.com/ZYM6-6/MCMIPL).
## Environment Settings
python: 3.8.0

pytorch: 1.8.1 

dgl: 0.8.1

## Training
`python RL_model.py --data_name <data_name>`

## Evaluation
`python evaluate.py --data_name <data_name> --load_rl_epoch <checkpoint_epoch>`

## Citation
If the code is used in your research, please star this repo and cite our paper as follows:
```
@inproceedings{zhao2023multi,
  title={Multi-view Hypergraph Contrastive Policy Learning for Conversational Recommendation},
  author={Zhao, Sen and Wei, Wei and Mao, Xian-Ling and Zhu, Shuai and Yang, Minghui and Wen, Zujie and Chen, Dangyang and Zhu, Feida},
  booktitle={Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={654--664},
  year={2023}
}


```

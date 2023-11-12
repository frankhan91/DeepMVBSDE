<div align="center">

# Learning high-dimensional McKean-Vlasov forward-backward stochastic differential equations with general distribution dependence

Jiequn Han, Ruimeng Hu, Jihao Long

[![arXiv](https://img.shields.io/badge/arXiv-2204.11924-b31b1b.svg)](https://arxiv.org/abs/2204.11924)


Link to code repository: https://github.com/frankhan91/DeepMVBSDE

</div>


## Dependencies
* Quick installation of conda environment for Python: ``conda env create -f environment.yml``

## Running
### Quick start for 2d version of the benchmark example with explicit solution in Section 4.1 of [paper](#Citation):
```
python main.py
```
You should be able to observe the program beginning its training within one minute. The total runtime is approximately 800 seconds on a MacBook Pro equipped with a 2.40 GHz Intel Core i9 processor. The configuration employs small parameters and short iterations, intended solely for testing purposes.

### To run experiments related to Figure 1 in [paper](#Citation)
```
python main.py --config_path configs/sinebm_d10.json
```
The experiments for d=5 can be conducted by setting `dim` to `5` in `eqn_config`. To employ the DBDP method for solving BSDEs, modify the `loss_type` field in `net_config`, changing it from `DeepBSDE` to `DBDPiter`.

### To run experiments related to Figure 2 in [paper](#Citation), for d=15
```
python main.py --config_path configs/sinebm_d15.json
```
To run solvers with other dimensions: d=5, 8, 10, 12, change `N_simu` and `N_simu` to 500, 800, 1000, 1200, respectively, and change `num_hiddens` in both `eqn_config` and `net_config` to [12, 12], [18, 18], [24, 24], [30, 30], respectively.

### To solve MFG of Cucker-Smale flocking model in Section 4.2 of [paper](#Citation)
```
python main.py --config_path configs/flock_d3.json
```

## Citation
[Han, Jiequn, Ruimeng Hu, and Jihao Long. *Learning high-dimensional McKean-Vlasov forward-backward stochastic differential equations with general distribution dependence*. arXiv preprint arXiv:2204.11924, accepted to SIAM Journal on Numerical Analysis (2022)](https://arxiv.org/abs/2204.11924)


If you find this work helpful, please consider starring this repo and citing our paper using the following Bibtex.
```bibtex
@article{HanHuLong2022deepmvbsde,
  title={Learning high-dimensional McKean-Vlasov forward-backward stochastic differential equations with general distribution dependence},
  author={Han, Jiequn and Hu, Ruimeng and Long, Jihao},
  journal={arXiv preprint arXiv:2204.11924, accepted to SIAM Journal on Numerical Analysis},
  year={2022}
}
```

## Contact
Please contact us at jiequnhan@gmail.com if you have any questions.

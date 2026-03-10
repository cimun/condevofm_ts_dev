# CondEvo for materials (condevofm)

This repository contains the code for the __CondEvo for materials (condevofm)__ package, an evolutionary framework for multi-modal atomistic structure prediction. 

It extends the [CondEvo](https://github.com/bhartl/CondEvo) package by Hartl et al. to make it employable for applications in materials science.

## Project

__TODO__

## Installation

```bash
conda create -n condevo-for-materials python=3.12 ipython
conda activate condevo-for-materials
git clone https://github.com/bhartl/foobench.git
pip install -e foobench/
git clone https://github.com/bhartl/CondEvo.git
pip install -e CondEvo/
git clone https://gitlab.tuwien.ac.at/e165-03-1_theoretische_materialchemie/madsen-s-research-group/condevo-for-materials.git
pip install -e condevo-for-materials/
```

## Examples

__TODO__

## Citation

If you use this code in your research, please cite the following paper:

```
@article{Hartl2024HADES,
      title={Heuristically Adaptive Diffusion-Model Evolutionary Strategy}, 
      author={Benedikt Hartl and Yanbo Zhang and Hananel Hazan and Michael Levin},
      journal={Advanced Science, in press},
      doi={10.1002/advs.202511537},
      year={2026},
      note = {Code available at \url{https://github.com/bhartl/condevo}; Preprinted as arxiv:2411.13420 (2024)},
}
```

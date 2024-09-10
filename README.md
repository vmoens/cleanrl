# LeanRL - Turbo-implementations of CleanRL scripts 

LeanRL is a fork of CleanRL, where selected PyTorch scripts optimized for performance.
The goal is to provide guidance on how to run your RL script at a speed that fully leverages PyTorch 2.0 and related
tooling while not compromising on the PyTorch API that made its success.We leverage `torch.compile` and `cudagraphs`
to achieve an order of magnitude speedup in training times.

## Key Features:
* 📜 Single-file implementation
   * We stick to the original spirit of CleanRL which is to keep *every detail about an algorithm variant in a single standalone file.* 
* 🚀 Fast implementations:
  * We provide an optimized, lean version of the PyTorch script where data copies and code execution have been
    optimized thanks to four tools: 
    * 🖥️ `torch.compile` to reduce the overhead and fuse operators whenever possible;
    * 📈 `cudagraphs` to isolate all the cuda operations and eliminate the cost of entering the compiled code;
    * 📖 `tensordict` to speed-up and clarify data copies on CUDA, facilitate functional calls and fast target parameters updates.
    * 🗺️ `torch.vmap` to vectorize the execution of the Q-value networks, when needed.
  * We provide a cleaned version of each script removing a bunch of LoC such as logging
    of certain values to focus on the time spent optimizing the model.
  * If available, we do the same with the Jax version of the code.
* 🪛 Local Reproducibility via Seeding

**Disclaimer**: This repo is a highly simplified version of CleanRL that lacks many features such as logging or
checkpointing - its only purpose is to provide various versions of similar training scripts to measure the plain
runtime under various constraints.

## Get started

Prerequisites:
* Python >=3.7.1,<3.11
- `pip install -r requirements/requirements.txt` for basic requirements, or another `.txt` file for specific applications.
- Upgrade torch to its nightly builds for a better coverage of `torch.compile`:
  - CUDA 11.8: `pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu118`
  - CUDA 12.1: `pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121`
  - CUDA 12.4: `pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124`
  - CPU: `pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu`

## Provided scripts

As of today, LeanRL focuses on three families of algorithms which we deem sufficient to give a didactic flavor of what
can be achieved with SOTA PyTorch programming:
- PPO (discrete - Atari)
- SAC (continuous)
- TD3 (continuous)

More algorithms should be coming soon.

Each algorithm under `./leanrl` has the same prefix (`ppo_atari_*.py`) and the suffix indicates what optimization have
been undergone (if any). The `torchcompile` versions are the main focus of this repo.

```bash
git clone https://github.com/pytorch-labs/leanrl.git && cd leanrl
<install deps here>

python leanrl/ppo_atari_torchcompile.py \
    --seed 1 \
    --total-timesteps 50000

```

## Citing CleanRL

LeanRL does not have a citation yet, credentials should be given to CleanRL instead.
To cite CleanRL in your work, please cite our technical [paper](https://www.jmlr.org/papers/v23/21-1342.html):

```bibtex
@article{huang2022cleanrl,
  author  = {Shengyi Huang and Rousslan Fernand Julien Dossa and Chang Ye and Jeff Braga and Dipam Chakraborty and Kinal Mehta and João G.M. Araújo},
  title   = {CleanRL: High-quality Single-file Implementations of Deep Reinforcement Learning Algorithms},
  journal = {Journal of Machine Learning Research},
  year    = {2022},
  volume  = {23},
  number  = {274},
  pages   = {1--18},
  url     = {http://jmlr.org/papers/v23/21-1342.html}
}
```


## Acknowledgement

CleanRL is a community-powered by project and our contributors run experiments on a variety of hardware.

* We thank many contributors for using their own computers to run experiments
* We thank Google's [TPU research cloud](https://sites.research.google/trc/about/) for providing TPU resources.
* We thank [Hugging Face](https://huggingface.co/)'s cluster for providing GPU resources. 

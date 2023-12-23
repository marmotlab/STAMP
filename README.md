# STAMP - Spatio-Temporal Attention network for Monitoring Persistently

This repository hosts the code for paper [Spatio-Temporal Attention Network for Persistent Monitoring of Multiple Mobile Targets](https://arxiv.org/abs/2303.06350), accepted for presentation at IROS 2023.

## Run

### Requirements
```bash
python >= 3.9
pytorch >= 1.11
ray >= 2.0
ortools
scikit-image
scikit-learn
scipy
imageio
tensorboard
```

### Training
1. Set appropriate parameters in `arguments.py -> Arguments`.
2. Run `python driver.py`.

### Evaluation
1. Set appropriate parameters in `arguments.py -> ArgumentsEval`.
2. Run `python /evals/eval_driver.py`.

## Files
- `arguments.py`: Training and evaluation arguments.
- `driver.py`: Driver of training program, maintain and update the global network.
- `runner.py`: Wrapper of the local network.
- `worker.py`: Interact with environment and collect episode experience.
- `network.py`: Spatio-temporal network architecture.
- `env.py`: Persistent monitoring environment.
- `gaussian_process.py`: Gaussian processes (wrapper) for belief representation.
- `/evals/*`: Evaluation files.
- `/utils/*`: Utility files for graph, target motion, and TSP.
- `/model/*`: Trained model.

### Demo

<img src="utils/media/demo.gif" alt="demo" style="width: 70%;">

<div>
    <h3><a href="https://youtu.be/q1wQup70m6c">Watch AirSim Video</a></h3>
    <a href="https://youtu.be/q1wQup70m6c">
        <img src="https://img.youtube.com/vi/q1wQup70m6c/maxresdefault.jpg" alt="Watch the video" style="width: 70%;">
    </a>
</div>




## Cite

```bibtex
@inproceedings{wang2023spatio,
  title={Spatio-Temporal Attention Network for Persistent Monitoring of Multiple Mobile Targets},
  author={Wang, Yizhuo and Wang, Yutong and Cao, Yuhong and Sartoretti, Guillaume},
  booktitle={2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2023}
}
```
Authors:
[Yizhuo Wang](https://github.com/wyzh98),
[Yutong Wang](https://github.com/wyt2019suzhou),
[Yuhong Cao](https://github.com/caoyuhong001),
[Guillaume Sartoretti](https://github.com/gsartoretti)

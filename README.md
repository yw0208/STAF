# STAF: 3D Human Mesh Recovery from Video with Spatio-Temporal Alignment Fusion
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/camenduru/STAF-colab/blob/main/STAF_colab.ipynb)
[![report](https://img.shields.io/badge/Project-Page-blue)](https://yw0208.github.io/staf/)
[![report](https://img.shields.io/badge/ArXiv-Paper-red)](https://arxiv.org/abs/2401.01730)

<p float="center">
  <video src="docs/demo_video.mp4" width="100%" />
</p>

## Getting Started 

#### Installation & Clone the repo [Environment on Linux (Ubuntu 18.04 with python >= 3.7)]

```bash
# Install the requirements using `virtualenv`: 
cd $PWD/STAF
source scripts/install_pip.sh
```

## Download the Required Data 

You can download the required data and the pre-trained STAF model from [here](https://drive.google.com/file/d/1PNrnAbnQ52jmddfFVKGr08CszVwHu0q9/view?usp=sharing). 
You need to unzip the contents and the data directory structure should follow the below hierarchy.

```
${ROOT}  
|-- data  
|   |-- base_data    
```

## Running the Demo

We have prepared a demo code to run STAF on arbitrary videos. 
To do this you can just run:

```bash
python demo.py --vid_file demo_video.mp4 --gpu 0
```
## Acknowledgments

Part of the code is borrowed from the following projects, including [PyMAF](https://github.com/HongwenZhang/PyMAF), [MPS-Net](https://github.com/MPS-Net/MPS-Net_release). Many thanks to their contributions.
Special thanks to [camenduru](https://github.com/camenduru/STAF-colab) for Colab Demo!




# STAF: 3D Human Mesh Recovery from Video with Spatio-Temporal Alignment Fusion

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



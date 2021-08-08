# Implementation of CLIP from OpenAI on a Web
##### This project was carried out as a small project from [SMARCLE](https://www.smarcle.dev/) and was produced with [Yoo Ji-won](https://github.com/Jiyajiwon).
## Installation
Needs Anaconda and NVIDIA GPU(not mandatory)
```
$ git clone https://github.com/kochanha/CLIP_Web_Project.git
$ cd CLIP_Web_Project
$ conda env create --file environment.yaml
$ pip install git+https://github.com/openai/CLIP.git
```
## Run Model
```
$ conda activate clip
$ python flask_main.py
```

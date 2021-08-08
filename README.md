# Implementation of [CLIP](https://github.com/openai/CLIP) from OpenAI on a Web
### This project was carried out as a small project from [SMARCLE](https://www.smarcle.dev/) and was produced with [Yoo Ji-won](https://github.com/Jiyajiwon).
![image](https://user-images.githubusercontent.com/44921488/128636325-6f14ca58-bfa3-4f35-94dd-8ff6eca6c11b.png)

## Installation
Needs Anaconda and NVIDIA GPU(not mandatory)
```
$ git clone https://github.com/kochanha/CLIP_Web_Project.git
$ cd CLIP_Web_Project
$ conda env create --file environment.yaml
$ conda activate clip
$ pip install git+https://github.com/openai/CLIP.git
```
## Run Model
```
$ python flask_main.py
```
## Project Pipeline
![그림1](https://user-images.githubusercontent.com/44921488/128636403-fea6929d-577b-4d56-a88c-0027539b2192.png)

### Motivation
All credits to Udemy course.

I converted from notebook to native python, it works well, split out learning portion to facilitate different movie length generation and decided to share.

creative_dream features:
* run init to check out the two images merging, early swap as you wish
* run model_train for as long as you wanted for video, change max_count or ctr-c
* run video to burn all computer dreamt frames into a video!

### Install Guide
copy and run (assume linux with python3):
```
python3 -m venv test
cd test
source bin/activate
git clone https://github.com/wendy-py/tensorflow_dream.git
cd tensorflow_dream
pip install tensorflow matplotlib opencv-python
```
to see the effects of two images coming together
```
python init.py
```
to generate frames of computer dream
```
python model_train.py
```
to generate a continuous video of available frames
```
python video.py
```

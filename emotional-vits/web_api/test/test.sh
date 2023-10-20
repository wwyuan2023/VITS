#!/bin/bash

# get
#wget -O ./a.wav 'http://127.0.0.1:10009/api/text2speech?tex="这是一个测试用例。"&vo=15&spd=12&per=678'

# post
#curl -o ./a.wav -v -X POST "http://127.0.0.1:10009/api/text2speech" -d '{"tex": "这是一个测试用例。", "per": 1, "vol": 15, "spd": 5, "pit": 5}' -H 'Content-type: application/json' -H 'accept: application/json'


python3 test_tts2.py 5000 10
python3 test_tts2.py 5000 12
python3 test_tts2.py 5000 20








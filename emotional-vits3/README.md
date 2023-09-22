# Emotional VITS
参考代码：https://github.com/innnky/emotional-vits
数据集无需任何情感标注，通过[情感提取模型](https://github.com/audeering/w2v2-how-to) 提取语句情感embedding输入网络，实现情感可控的VITS合成


## 1. 安装依赖
```
pip install -r requirments.txt
pip install monotonic-align
pip install textparser-${lastest-version}.tar.gz
```

## 2. 数据准备
```
# 参考数据目录dataset
# dataset/
# ├──wav/*.wav
# ├──txt/*.lab
# ├──spkid.txt
# ├──train.scp
# └──valid.scp

# 生成文本vector
outdir=dataset/vec
mkdir $outdir
cat dataset/txt/*.lab | text-vectorization -o $outdir

# 提取情感向量
outdir=dataset/emo
mkdir $outdir
find dataset/wav -name "*.wav" > files.scp
python3 toolkits/extract_emotion.py --scp files.scp --outdir $outdir
rm -f files.scp
# 注意：首次运行会自动下载模型到~/.cache/huggingface/

# 生成filelist
# 参考filelists目录下的train.scp和valid.scp
# 格式：vecfn|wavfn|emofn|spkid
```

## 3. 开始训练
```
python train.py -c configs/base.json -m ${your_model_name}
```

## 4. 导出模型
```
# 注意 checkpoint/ 是模型默认的存储目录，保存ckpt和emo文件
# 导出ckpt
python export.py --ckpt logs/${your_model_name}/G_${num}.pth --outdir ./checkpoint/
# 导出emo，每个发音人都有自己的emo聚类，根据每个发音人风格和录音数目，选择K的大小
K=7
python toolkits/cluster_emotion.py $K emo-list-scp checkpoint/${spkid}.emo
```

## 5. 部署模型
```
cd web_api
pip install -r requirments.txt
pip install textparser-*.tar.gz
sh start.sh # 开启tts服务
```





tts_server_ip=36.139.230.78
tts_server_ip=36.139.229.113
tts_server_ip=127.0.0.1


# 增加发音人：POST上传发音人的录音和文本
url=http://${tts_server_ip}:10010/api/sat/uploadfile/${spkid}

# 删除发音人：存储在服务端的用户录音和文本数据会被清空
curl -v -X POST http://${tts_server_ip}:10010/api/sat/clean/${spkid}

# 查询发音人：data字段返回发音人信息
# 例如："data":{"10010":20,"10015":12,"10077":101}
# key/value分别是spkid和录音个数
curl -v -X POST http://${tts_server_ip}:10010/api/sat/spkinfo


# 开始训练，注意：开始训练的时候，服务端会自动将tts server关闭；
# 开启训练时，所有的发音人都会参与训练
# 增加或删除发音人都需要重新训练
curl -v -X POST http://${tts_server_ip}:10010/api/sat/start

# 查询训练状态
curl -v -X POST http://${tts_server_ip}:10010/api/sat/status


# 结束训练，注意如果在训练中途中断训练，可能tts server无法自动重启，需要调用下面的接口开启tts server的
curl -v -X POST http://${tts_server_ip}:10010/api/sat/stop

# 开启tts server
curl -v -X POST http://${tts_server_ip}:10010/api/sat/start/tts



# coding: utf-8

import os, sys
import time
import requests
import json
import threading
import numpy as np

# POST方法调用tts
def post(id, url, tex, per):
    data = {
        "tex" : tex,
        "per" : per,
    }
    response = requests.post(url, data=json.dumps(data))
    content = response.content
    #with open("out.wav", "wb") as f:
    #    f.write(content)
    response.close()
    return {"id": id, "content": len(content)}

# 替我们工作的线程池中的线程
class MyThread(threading.Thread):
    def __init__(self, workList, resultList, **kwargs):
        threading.Thread.__init__(self, kwargs=kwargs)
        # 线程在结束前等待任务队列多长时间
        self.daemon = True
        self.workList = workList
        self.resultList = resultList
        #self.start()

    def run(self):
        while len(self.workList) > 0:
            try:
                # 从工作队列中获取一个任务
                callable, args, kwargs = self.workList.pop(0)
                # 我们要执行的任务
                start = time.time()
                result = callable(*args, **kwargs)
                # 报任务返回的结果放在结果队列中
                result["name"] = self.name
                result["start"] = start
                result["end"] = time.time() # 任务结束时间
                self.resultList.append(result)
            except :
                print(sys.exc_info())
                raise

class ThreadPool:
    def __init__(self, num_of_threads=10):
        self.workList = []
        self.resultList = []
        self.threads = []
        for i in range(num_of_threads):
            self.workList.append([])
            self.resultList.append([])
            thread = MyThread(self.workList[-1], self.resultList[-1])
            self.threads.append(thread)
        self.idx = 0
        self.total = num_of_threads

    def wait_for_complete(self):
        # start 
        for th in self.threads:
            th.start()
        # 等待所有线程完成。
        while len(self.threads):
            thread = self.threads.pop()
            # 等待线程结束
            if thread.is_alive(): # 判断线程是否还存活来决定是否调用join
                thread.join()

    def add_job(self, callable, *args, **kwargs):
        self.workList[self.idx].append([callable, args, kwargs])
        self.idx += 1
        self.idx = self.idx % self.total


def main():
    
    url = "http://36.140.33.30:10009/api/text2speech"
    #url = "http://127.0.0.1:10009/api/text2speech"
    #url = "http://36.139.229.113:10009/api/text2speech"
    text = "这是一个测试用例这是一个测试用例这是一个测试用例这是一个测试用例这是一个测试用例这是一个测试用例这是一个测试用例这是一个测试。"
    spkid = 678
    
    NUM_TASK = int(sys.argv[1]) # 5000
    NUM_WORK = int(sys.argv[2]) # 10
    if len(sys.argv) >= 4:
        textfn = sys.argv[3]
        assert os.path.exists(textfn)
        with open(textfn, 'rt', encoding='utf-8') as f:
            text = [x.strip() for x in f]

    print("Start testing...")
    start = time.time()
    tp = ThreadPool(num_of_threads=NUM_WORK)
    for i in range(NUM_TASK):
        if type(text) is str:
            tp.add_job(post, i, url, text, spkid)
        else:
            tp.add_job(post, i, url, text[i % len(text)], spkid)
    tp.wait_for_complete()
    time_used = time.time() - start
    print("End testing!")
    print(f"Task={NUM_TASK}, Thread={NUM_WORK}, Time used={time_used:.03f} Sec")
    
    for th in tp.resultList:
        for ret in th:
            pass
            #print(ret)


if __name__ == "__main__":

    main()

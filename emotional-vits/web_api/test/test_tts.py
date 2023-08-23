# coding: utf-8

import os, sys
import time
import requests
import json
import threading
import queue
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
    def __init__(self, workQueue, resultQueue, timeout=30, **kwargs):
        threading.Thread.__init__(self, kwargs=kwargs)
        # 线程在结束前等待任务队列多长时间
        self.timeout = timeout
        self.daemon = True
        self.workQueue = workQueue
        self.resultQueue = resultQueue
        self.start()

    def run(self):
        while True:
            try:
                # 从工作队列中获取一个任务
                callable, args, kwargs = self.workQueue.get(timeout=self.timeout)
                # 我们要执行的任务
                start = time.time()
                ret = callable(*args, **kwargs)
                # 报任务返回的结果放在结果队列中
                ret["name"] = self.name
                ret["start"] = start
                ret["end"] = time.time() # 任务结束时间
                self.resultQueue.put(ret)
            except queue.Empty: # 任务队列空的时候结束此线程
                break
            except :
                print(sys.exc_info())
                raise

class ThreadPool:
    def __init__(self, num_of_threads=10):
        self.workQueue = queue.Queue()
        self.resultQueue = queue.Queue()
        self.threads = []
        self.__createThreadPool(num_of_threads)

    def __createThreadPool(self, num_of_threads):
        for i in range(num_of_threads):
            thread = MyThread(self.workQueue, self.resultQueue)
            self.threads.append(thread)

    def wait_for_complete(self):
        # 等待所有线程完成。
        while len(self.threads):
            thread = self.threads.pop()
            # 等待线程结束
            if thread.is_alive(): # 判断线程是否还存活来决定是否调用join
                thread.join()

    def add_job(self, callable, *args, **kwargs):
        self.workQueue.put( (callable, args, kwargs) )


def main():
    
    url = "http://36.140.33.30:10009/api/text2speech"
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
    tp = ThreadPool(num_of_threads=NUM_WORK)
    for i in range(NUM_TASK):
        if type(text) is str:
            tp.add_job(post, i, url, text, spkid)
        else:
            tp.add_job(post, i, url, text[i % len(text)], spkid)
    tp.wait_for_complete()
    print("End testing!")
    
    # 处理结果
    num_ret = tp.resultQueue.qsize()
    print(f"Result Queue's length == {num_ret}")
    time_used = []
    while tp.resultQueue.qsize():
        ret = tp.resultQueue.get()
        #print(ret)
        start, end = ret["start"], ret["end"]
        time_used.append(end - start)
    time_used = np.array(time_used, dtype=np.float32)
    time_used_mean, time_used_std = np.mean(time_used), np.std(time_used)
    time_used_min, time_used_max = np.min(time_used), np.max(time_used)
    time_used_median = np.median(time_used)
    print(f"Task={NUM_TASK}, Thread={NUM_WORK}, Time used: mean/std={time_used_mean:.03f}/{time_used_std:.03f}, min/max={time_used_min:.03f}/{time_used_max:.03f}, median={time_used_median:.03f}")


if __name__ == "__main__":

    main();

# encoding: utf-8
# https://fastapi.tiangolo.com/zh/tutorial/
#

import os
import time
import glob
from multiprocessing import Process
from fastapi import FastAPI, Form, UploadFile
from fastapi.responses import Response, JSONResponse, FileResponse


SAT_DIR, OUT_DIR = None, None
SATPID = None

app = FastAPI()


@app.on_event("startup")
async def startup_event():
    global SAT_DIR, OUT_DIR, SATPID
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    SAT_DIR = os.path.abspath(os.path.join(cur_dir, "../sat"))
    OUT_DIR = os.path.abspath(os.path.join(cur_dir, "../checkpoint/"))
    SATPID = None


@app.on_event("shutdown")
async def startup_event():
    global SAT_DIR, CKPT_PATH, OUT_DIR, SATPID
    SAT_DIR, CKPT_PATH, OUT_DIR, SATPID = None, None, None, None


@app.post("/api/sat/uploadfile/{spkid}")
async def sat_upload_file(spkid: int, file: UploadFile, text: str = Form()):
    global SAT_DIR

    if spkid < 10000:
        return JSONResponse(status_code=400, content={"code": 400, "data": {}, "msg": f"error: spkid={spkid} must more than 10000"})

    data_dir = os.path.join(SAT_DIR, f"data/{spkid}")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    data = await file.read()
    outfn = os.path.join(data_dir, file.filename)
    with open(outfn, "wb") as f:
        f.write(data)
    outfn = outfn.replace(".wav", ".txt")
    with open(outfn, "wt", encoding="utf-8") as f:
        f.write(text)
        f.write("\n")

    return JSONResponse(status_code=200, content={"code": 200, "data": {}, "msg": f"save file ok!"})


def _sat_clean(spkid: int):
    global SAT_DIR

    data_dir = os.path.join(SAT_DIR, f"data/{spkid}")
    if not os.path.exists(data_dir):
        return JSONResponse(status_code=400, content={"code": 400, "data": {}, "msg": f"error: there is no any data for spkid={spkid}"})

    if has_sat():
        return JSONResponse(status_code=400, content={"code": 400, "data": {}, "msg": f"error: sat is training"})

    cmd = f"rm -rf {data_dir}"
    os.system(cmd)
    return JSONResponse(status_code=200, content={"code": 200, "data": {}, "msg": f"sat clean success, spkid={spkid}"})

def _sat_spkinfo():
    global SAT_DIR

    data_dir = os.path.join(SAT_DIR, "data")
    if not os.path.exists(data_dir):
        return JSONResponse(status_code=400, content={"code": 400, "data": {}, "msg": f"error: there is no any speaker record data"})

    spkid = {}
    for spkdir in glob.glob(f"{data_dir}/*"):
        sid = os.path.basename(spkdir)
        if os.path.isdir(spkdir) and sid.isdigit():
            spkid[sid] = len(glob.glob(spkdir + "/*.wav"))
    if len(spkid) == 0:
        return JSONResponse(status_code=400, content={"code": 400, "data": {}, "msg": f"error: there is no any speaker record data"})

    return JSONResponse(status_code=200, content={"code": 200, "data": spkid, "msg": f"sat speaker number={len(spkid)}"})


@app.post("/api/sat/clean/{spkid}")
async def sat_clean(spkid: int):
    return _sat_clean(spkid)

@app.get("/api/sat/clean/{spkid}")
async def sat_clean(spkid: int):
    return _sat_clean(spkid)

@app.post("/api/sat/spkinfo")
async def sat_spkinfo():
    return _sat_spkinfo()

@app.get("/api/sat/spkinfo")
async def sat_spkinfo():
    return _sat_spkinfo()


def has_tts():
    cmd = "ps -ef | grep python | grep 'socket_server' | grep -v grep | awk '{print $2}'"
    ret = os.popen(cmd)
    ret = ret.read()
    if len(ret) > 0:
        return True
    return False

def stop_tts():
    for _ in range(10):
        cmd = "sh ./stop.sh"
        os.system(cmd)
        cmd = "ps -ef | grep python | grep 'socket_server' | grep -v grep | awk '{print $2}'"
        ret = os.popen(cmd)
        ret1 = ret.read()
        cmd = "ps -ef | grep python | grep 'http_server:app' | grep -v grep | awk '{print $2}'"
        ret = os.popen(cmd)
        ret2 = ret.read()
        if len(ret1) == 0 and len(ret2) == 0:
            break
        time.sleep(1)
        
def start_tts():
    global SAT_DIR, OUT_DIR
    ckpt1 = os.path.join(OUT_DIR, "checkpoint.pth")
    ckpt2 = os.path.join(SAT_DIR, "pretrain/G_0.pth")
    if not os.path.exists(ckpt1) and os.path.exists(ckpt2):
        conf1 = os.path.join(OUT_DIR, "config.json")
        conf2 = os.path.join(SAT_DIR, "configs/adapt.json")
        cmd = f"ln -sf -T {ckpt2} {ckpt1}; ln -sf -T {conf2} {conf1};"
        os.system(cmd)
    for _ in range(10):
        cmd = "sh ./start.sh"
        os.system(cmd)
        cmd = "ps -ef | grep python | grep 'socket_server' | grep -v grep | awk '{print $2}'"
        ret = os.popen(cmd)
        ret1 = ret.read()
        cmd = "ps -ef | grep python | grep 'http_server:app' | grep -v grep | awk '{print $2}'"
        ret = os.popen(cmd)
        ret2 = ret.read()
        if len(ret1) > 0 and len(ret2) > 0:
            break
        time.sleep(5)

def has_sat():
    global SATPID
    cmd = "ps -ef | grep ./adapt.sh | grep -v grep"
    ret = os.popen(cmd)
    ret = ret.read()
    if len(ret) > 0 or (SATPID is not None and SATPID.is_alive()):
        return True
    SATPID = None
    return False

def start_sat(work_dir, out_dir, logfn):

    # stop tts
    is_tts = has_tts()
    if is_tts: stop_tts()

    # start sat
    cmd = f"cd {work_dir}; sh ./adapt.sh --outdir {out_dir} >{logfn} 2>&1"
    os.system(cmd)

    # start tts
    if is_tts: start_tts()

def stop_sat():
    for _ in range(10):
        cmd = "ps -ef | grep ./adapt.sh | grep -v grep"
        ret = os.popen(cmd)
        ret = ret.read()
        if len(ret) == 0:
            break
        cmd  = "kill $(ps -ef | grep ./adapt.sh | grep -v grep | awk '{print $2}') "
        cmd += "$(ps -ef | grep train.py | grep -v grep | awk '{print $2}') "
        os.system(cmd)
        time.sleep(1)


def _sat_start_training():
    global SAT_DIR, OUT_DIR, SATPID

    if has_sat():
        return JSONResponse(status_code=400, content={"code": 400, "data": {}, "msg": f"error: sat is training"})

    logfn = os.path.join(SAT_DIR, "train.log")
    p = Process(target=start_sat, args=(SAT_DIR, OUT_DIR, logfn))
    p.daemon = True
    p.start()
    SATPID = p

    return JSONResponse(status_code=200, content={"code": 200, "data": {}, "msg": f"sat start training success!"})

def _sat_stop_training():
    global SATPID
    stop_sat()
    SATPID = None
    start_tts()
    return JSONResponse(status_code=200, content={"code": 200, "data": {}, "msg": f"sat stop training success!"})

def _sat_status():
    global SAT_DIR, OUT_DIR
    
    if has_sat():
        return JSONResponse(status_code=200, content={"code": 201, "data": {}, "msg": f"sat is training!"})
    
    ckpt = os.path.join(OUT_DIR, "checkpoint.pth")
    data_dir = os.path.join(SAT_DIR, "data")
    spkid1 = sorted([ os.path.basename(spkdir) for spkdir in glob.glob(f"{data_dir}/*") ])
    spkid2 = sorted([ os.path.splitext(os.path.basename(sparam))[0] for sparam in glob.glob(f"{OUT_DIR}/*.emo") ])
    if not os.path.exists(ckpt) or any([x1 not in spkid2 for x1 in spkid1]):
        return JSONResponse(status_code=200, content={"code": 202, "data": {}, "msg": f"sat training failure!"})

    return JSONResponse(status_code=200, content={"code": 200, "data": {}, "msg": f"sat training success!"})

def _sat_start_tts():
    if has_sat():
        return JSONResponse(status_code=400, content={"code": 400, "data": {}, "msg": "error: sat is training!"})
    start_tts()
    if has_tts():
        return JSONResponse(status_code=200, content={"code": 200, "data": {}, "msg": "start tts success!"})
    return JSONResponse(status_code=400, content={"code": 400, "data": {}, "msg": "error: start tts failure!"})


@app.post("/api/sat/start")
async def sat_start_training():
    return _sat_start_training()

@app.get("/api/sat/start")
async def sat_start_training():
    return _sat_start_training()

@app.post("/api/sat/stop")
async def sat_stop_training():
    return _sat_stop_training()

@app.get("/api/sat/stop")
async def sat_stop_training():
    return _sat_stop_training()

@app.post("/api/sat/status")
async def sat_status():
    return _sat_status()

@app.get("/api/sat/status")
async def sat_status():
    return _sat_status()

@app.post("/api/sat/start/tts")
async def sat_start_tts():
    return _sat_start_tts()

@app.get("/api/sat/start/tts")
async def sat_start_tts():
    return _sat_start_tts()

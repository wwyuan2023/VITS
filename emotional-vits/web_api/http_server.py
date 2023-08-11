# encoding: utf-8
# https://fastapi.tiangolo.com/zh/tutorial/
#

import os
import socket
import numpy as np

from typing import Union, List
from fastapi import FastAPI, Query, Body, UploadFile
from fastapi.responses import Response, JSONResponse, FileResponse
from pydantic import BaseModel, constr, conint, confloat, conlist

from socket_client import synthesize

TCP_SOCKET_CLIENT = None
REMOTE = ('127.0.0.1', 5959)

def connect_tts_server():
    global TCP_SOCKET_CLIENT, REMOTE
    if TCP_SOCKET_CLIENT is None or getattr(TCP_SOCKET_CLIENT, "_closed"):
        tcp_client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcp_client_socket.settimeout(10)
        if 0 != tcp_client_socket.connect_ex(REMOTE):
            tcp_client_socket.close()
            del tcp_client_socket
            tcp_client_socket = None
        TCP_SOCKET_CLIENT = tcp_client_socket
    if TCP_SOCKET_CLIENT is None:
        return False
    return True

def disconnect_tts_server():
    global TCP_SOCKET_CLIENT
    if TCP_SOCKET_CLIENT is not None:
        if not getattr(TCP_SOCKET_CLIENT, "_closed"):
            TCP_SOCKET_CLIENT.close()
        del tcp_client_socket
    TCP_SOCKET_CLIENT = None
    
def reconnect_tts_server():
    disconnect_tts_server()
    connect_tts_server()


app = FastAPI()


@app.on_event("startup")
async def startup_event():
    connect_tts_server()

@app.on_event("shutdown")
async def startup_event():
    disconnect_tts_server()


@app.get("/api/text2speech")
async def text2speech(
    tex: str = Query(..., min_length=1, max_length=1024),
    per : int = Query(678, ge=1),
    vol : int = Query(15, ge=0, le=15),
    spd: int = Query(5, ge=0, le=15),
    pit: int = Query(5, ge=0, le=15),
    emo: int = Query(0, ge=0)
):
    global TCP_SOCKET_CLIENT, REMOTE
    if not connect_tts_server() and not connect_tts_server():
        return {"msg": f"can not connect tts server{REMOTE}\n"}
    emotion = (emo, -1)
    inputs = {
        "text": tex,
        "spkid": per,
        "volume": vol/15.0,
        "speed": 2**(-(spd-5)/12.0),
        "pitch": 2**((pit-5)/12.0),
        "emotion": emotion,
    }
    outputs, TCP_SOCKET_CLIENT = synthesize(inputs, REMOTE, TCP_SOCKET_CLIENT, True)
    if outputs is None:
        # try again, due to python socket cannot check alive
        reconnect_tts_server()
        outputs, TCP_SOCKET_CLIENT = synthesize(inputs, REMOTE, TCP_SOCKET_CLIENT, True)
    if outputs is None:
        disconnect_tts_server()
        return {"msg": f"synthesis failure!"}
    wav = outputs.pop('wav')
    return Response(content=wav, media_type="audio/wav") 


class InputBody(BaseModel):
    tex: constr(min_length=1, max_length=100*1024)
    per: conint(ge=0) = 678
    vol: conint(ge=0, le=15) = 15
    spd: conint(ge=0, le=15) = 5
    pit: conint(ge=0, le=15) = 5
    emo: Union[int, List[float]] = 0

@app.post("/api/text2speech")
async def text2speech(inputbody: InputBody):
    global TCP_SOCKET_CLIENT, REMOTE
    if isinstance(inputbody.emo, int):
        emotion = (inputbody.emo, -1)
    elif len(inputbody.emo) != 1024:
        return {"msg": f"emo must be int or list[float] with length=1024 !"}
    else:
        emotion = (np.array(inputbody.emo).astype(np.float32), -1)
    if not connect_tts_server() and not connect_tts_server():
        return {"msg": f"can not connect tts server{REMOTE}\n"}
    inputs = {
        "text": inputbody.tex,
        "spkid": inputbody.per,
        "volume": inputbody.vol/15.0,
        "speed": 2**(-(inputbody.spd-5)/12.0),
        "pitch": 2**((inputbody.pit-5)/12.0),
        "emotion": emotion,
    }
    outputs, TCP_SOCKET_CLIENT = synthesize(inputs, REMOTE, TCP_SOCKET_CLIENT, True)
    if outputs is None:
        # try again, due to python socket cannot check alive
        reconnect_tts_server()
        outputs, TCP_SOCKET_CLIENT = synthesize(inputs, REMOTE, TCP_SOCKET_CLIENT, True)
    if outputs is None:
        disconnect_tts_server()
        return {"msg": f"synthesis failure!"}
    wav = outputs.pop('wav')
    return Response(content=wav, media_type="audio/wav")



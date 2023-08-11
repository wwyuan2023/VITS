# encoding: utf-8

import os, sys
import socket
import struct, pickle, json
import time
import argparse


def synthesize(inputs, remote=('localhost', 5959), tcp_client_socket=None, return_socket=False):
    
    try:
        if tcp_client_socket is None:
            tcp_client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            tcp_client_socket.settimeout(60)
            tcp_client_socket.connect_ex(remote)
        data = pickle.dumps(inputs)
        data_size = sys.getsizeof(data)
        tcp_client_socket.send(struct.pack('i', data_size))
        tcp_client_socket.sendall(data)
        data = tcp_client_socket.recv(4)
        data_size = struct.unpack('i', data)[0]
        body = b""
        while sys.getsizeof(body) < data_size:
            body += tcp_client_socket.recv(data_size)
        outputs = pickle.loads(body)
    except Exception as e:
        print("synthesize:: Exception:", e)
        outputs = None
    finally:
        if not return_socket or outputs is None:
            tcp_client_socket.close()
            del tcp_client_socket
            tcp_client_socket = None
    
    items = outputs if not return_socket else (outputs, tcp_client_socket)
    return items

if __name__ == "__main__":

    inputs = {
        "text": "这是1个测试用例。this is a test sample.",
        "spkid": 674,
        "volume": 1.0,
        "speed": 1.0,
        "pitch": 1.0,
    }
    print("To synthesize:\n", inputs)
    outputs = synthesize(inputs)
    if outputs is None:
        print("Synthesis failure!")
    
    wav = outputs.pop('wav')
    dur = outputs.pop('dur')
    print(outputs)
    with open('./a.wav', 'wb') as f:
        f.write(wav)
    print("Done!")


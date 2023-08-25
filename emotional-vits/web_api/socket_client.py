# encoding: utf-8

import os, sys
import socket
import struct, pickle
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
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, required=False, default='127.0.0.1',
                        help='What TTServer IP to connect to. (default=localhost)')
    parser.add_argument('--port', type=int, required=False, default=5959,
                        help='Port to serve TTServer on. (default=5959)')
    parser.add_argument('--utterance', '-u', type=str, required=False,
                        help='Input utterance with UTF-8 encoding to synthesize.')
    parser.add_argument('--textfile', '-t', type=str, required=False,
                        help='Input text file with UTF-8 encoding to synthesize.')
    parser.add_argument('--spkid', '-i', type=int, required=False, default=1,
                        help='Set speaker ID. (default=1)')
    parser.add_argument('--volume', '-v', type=float, required=False, default=1.0,
                        help='Set volume, its range is (0.0, 1.0]. (default=1.0)')
    parser.add_argument('--speed', '-s', type=float, required=False, default=1.0,
                        help='Set speed, its range is (0.5, 1.0]. (default=1.0)')
    parser.add_argument('--pitch', '-p', type=float, required=False, default=1.0,
                        help='Set pitch, its range is (0.0, 1.0]. (default=1.0)')
    parser.add_argument('--sampling-rate', '-r', type=int, required=False,
                        help='Set sampling rate.')
    parser.add_argument('--outdir', '-o', type=str, required=True,
                        help='Directory for saving synthetic wav.')
    args = parser.parse_args()
    
    # check args
    if args.utterance is None or args.textfile is None:
        raise ValueError("Please specify either --utterance or --textfile")
    
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    
    # remote addr
    host, port = args.host, int(args.port)
    remote = (host, port)
    print(f"Remote = {remote}")
    
    # pack inputs
    inputs = {
        "spkid": args.spkid,
        "volume": args.volume,
        "speed": args.speed,
        "pitch": args.pitch,
    }
    
    utt_text = []
    if args.utterance is not None:
        utt_text.append(args.utterance)
    if args.textfile is not None:
        with open(args.textfile, 'rt') as f:
            for line in f:
                line = line.strip()
                if len(line) == 0: continue
                utt_text.append(line)
    
    # syntheize
    for idx, text in enumerate(utt_text, 1):
        inputs["text"] = text
        print("To synthesize:\n", inputs)
        outputs = synthesize(inputs)
        if outputs is None:
            print("Synthesis failure!")
            continue
    
        wav = outputs.pop('wav')
        print(outputs)
        with open(os.path.join(args.outdir, f"{idx:06d}.wav"), 'wb') as f:
            f.write(wav)
    
    print("Done!")


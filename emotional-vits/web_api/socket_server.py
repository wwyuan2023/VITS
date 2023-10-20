# encoding: utf-8

import os, sys
import socket
import struct, pickle, json
import time
import argparse
import torch

import multiprocessing as mp


def strftime(): 
    return time.strftime('%Y-%m-%d %H:%M:%S')


def tts_proc(tcp_server_socket, buffer_size, work, rank, loglv=0):
    sys.path.insert(0, os.path.abspath('..'))
    from vits_wrap import VITSWrap
    #torch.cuda.set_device(rank)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device('cpu')

    func_name = f"tts_proc: work{work}(pid={os.getpid()})"
    print(f"{strftime()} {func_name}, load tts model@{device} ...")
    tts = VITSWrap(device=device)
    print(f"{strftime()} {func_name}, load tts model@{device} done!")
    print(f"{strftime()} {func_name}, ckpt root={tts.speecher.res_root_path}")

    while True:
        print(f"{strftime()} {func_name}, waiting be connected ...")
        conn, addr = tcp_server_socket.accept()
        print(f"{strftime()} {func_name}, waiting to recieve data from client({addr}) ... ")
        try:
            while True:
                # recive data
                data = conn.recv(4)
                if not data:
                    print(f"{strftime()} {func_name}, client disconnection!")
                    break
                data_size = struct.unpack('i', data)[0]
                if data_size == 0:
                    print(f"{strftime()} {func_name}, recv data size={data_size},"
                          "client socket will be closed by server!")
                    break
                elif data_size > buffer_size:
                    print(f"{strftime()} {func_name}, recv data size={data_size} > {buffer_size},"
                          "serving refused, client socked will be closed by server!")
                    break
                else:
                    if loglv > 0:
                        print(f"{strftime()} {func_name}, recv data size={data_size}")
                body = b""
                while sys.getsizeof(body) < data_size:
                    body += conn.recv(data_size)
                inputs = pickle.loads(body)
                if loglv > 0:
                    print(f"{strftime()} {func_name}, recived client data:\n{json.dumps(inputs, indent=4, ensure_ascii=False)}")
                
                # synthesis
                outputs = tts.speaking(inputs)

                # sent outputs
                body = pickle.dumps(outputs)
                data_size = sys.getsizeof(body)
                conn.send(struct.pack('i', data_size))
                conn.sendall(body)

                # update state
                tts.update()

                # statistic
                rtf = outputs.get('rtf', 1.0)
                segtext = outputs.get('segtext', None)
                if loglv > 0:
                    print(f"{strftime()} {func_name}, sent synthesis results to client={addr}, data size={data_size}, rtf={rtf:.3f}, segtext=`{segtext}`")

                # clear up
                del outputs
                
        except Exception as e:
            print(f"{strftime()} {func_name}, Exception: ", str(e))
        finally:
            conn.close()
            del conn
            print(f"{strftime()} {func_name}, client socket {addr} is closed!")
        if getattr(tcp_server_socket, "_closed"):
            break
    del tts
    print(f"{strftime()} {func_name}, unload tts model@{device}")
    return


class TTServer:
    def __init__(
        self,
        host="127.0.0.1",
        port=5959,
        num_gpus=1,
        max_jobs_per_gpu=1,
        max_input_size=256,
        loglv=0,
    ):
        self.loglv = loglv
        self.host = host
        self.port = port
        self.num_gpus = num_gpus
        self.max_jobs_per_gpu = max_jobs_per_gpu
        self.max_input_size = max_input_size
        
        self.ps = []
    
    def run(self):
        func_name = "TTServer::run"
        try:
            tcp_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            tcp_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            print(f"{func_name}: tcp server socket is created!")
            tcp_server_socket.bind((self.host, self.port))
            print(f"{func_name}: bind on host={self.host}, port={self.port}")
            tcp_server_socket.listen(1)
            work = 0
            for rank in range(self.num_gpus):
                for _ in range(self.max_jobs_per_gpu):
                    self.ps += [
                        mp.Process(target=tts_proc, args=(tcp_server_socket, self.max_input_size, work, rank, self.loglv))
                    ]
                    self.ps[-1].daemon = True
                    self.ps[-1].start()
                    print(f"{func_name}: start tts process ...")
            for p in self.ps:
                p.join()
        except Exception as e:
            print(f"{func_name}, Exception", str(e))
        finally:
            tcp_server_socket.close()
            print(f"{func_name}: tcp server socket is closed!")
            for p in self.ps:
                if p.is_alive(): p.terminate()
            for p in self.ps:
                p.join()
        return


if __name__ == "__main__":

    mp.set_start_method("spawn")
    num_gpus, num_jobs = 1, 1
    #num_gpus = torch.cuda.device_count()
    #assert num_gpus, "Requires CUDA!"
    max_input = 1 * 100 * 1024
    loglv = 0

    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, required=False, default='127.0.0.1',
                        help='What host IP to listen to. (default=localhost)')
    parser.add_argument('--port', type=int, required=False, default=5959,
                        help='Port to serve TTServer on. Pass 0 to request an unused port selected by the operation system. (default=5959)')
    parser.add_argument('--n-gpus', '-n', type=int, required=False, default=num_gpus,
                        help='Number of GPUs to used. (default={})'.format(num_gpus))            
    parser.add_argument('--n-jobs', '-j', type=int, required=False, default=num_jobs,
                        help='Number of parallel processes per GPU. (default={})'.format(num_jobs))
    parser.add_argument('--max-input', '-m', type=int, required=False, default=max_input,
                        help='Maximum byte size of input. (default={})'.format(max_input))
    parser.add_argument('--loglv', '-d', type=int, required=False, default=loglv,
                        help='Log level. (default={})'.format(loglv))
    args = parser.parse_args()
    
    ttsd = TTServer(args.host, int(args.port), int(args.n_gpus), int(args.n_jobs), int(max_input), int(args.loglv))
    ttsd.run()


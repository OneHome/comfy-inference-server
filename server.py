import threading
import traceback
import queue
import re
import requests
import os
import torch
import sys
import datetime
import argparse
import time
import random
import tqdm

import websockets.exceptions
import websockets.sync.server
from bson import encode as bson_encode, decode as bson_decode

import secrets
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.exceptions import InvalidTag


DEFAULT_PASSWORD = "qDiffusion"
FRAGMENT_SIZE = 1048576
UPLOAD_IDS = {}

def log_traceback(label):
    exc_type, exc_value, exc_tb = sys.exc_info()
    tb = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    with open("crash.log", "a", encoding='utf-8') as f:
        f.write(f"{label} {datetime.datetime.now()}\n{tb}\n")
    print(label, tb)
    return tb

def get_scheme(password):
    password = password.encode("utf8")
    h = hashes.Hash(hashes.SHA256())
    h.update(password)
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=h.finalize()[:16],
        iterations=480000,
    )
    return AESGCM(kdf.derive(password))

def encrypt(scheme, data):
    if scheme:
        nonce = secrets.token_bytes(16)
        data = nonce + scheme.encrypt(nonce, data, b"")
    return data

def decrypt(scheme, data):
    if scheme:
        data = scheme.decrypt(data[:16], data[16:], b"")
    return data

def get_id():
    return random.SystemRandom().randint(1, 2**31 - 1)

SEP = os.path.sep
INV_SEP = {"\\": '/', '/':'\\'}[os.path.sep]
NO_CONV = {"prompt", "negative_prompt", "url", "trace"}

def convert_path(p):
    return p.replace(INV_SEP, SEP)

def convert_all_paths(j):
    if type(j) == list:
        for i in range(len(j)):
            v = j[i]
            if type(v) == str and INV_SEP in v:
                j[i] = convert_path(v)
            if type(v) == list or type(v) == dict:
                convert_all_paths(j[i])
    elif type(j) == dict: 
        for k, v in j.items():
            if k in NO_CONV: continue
            if type(v) == str and INV_SEP in v:
                j[k] = convert_path(v)
            if type(v) == list or type(v) == dict:
                convert_all_paths(j[k])

def _download(url, path, progress_callback=None, started_callback=None, headers={}):
    if os.path.isdir(path):
        filename = None
        folder = path
    else:
        filename = path
        folder = os.path.dirname(path)
        os.makedirs(folder, exist_ok=True)

    resp = requests.get(url, stream=True, timeout=10, headers=headers, allow_redirects=True)
    total = int(resp.headers.get('content-length', 0))

    content_length = resp.headers.get("content-length", 0)
    if not content_length:
        raise RuntimeError(f"response is empty")

    content_type = resp.headers.get("content-type", "unknown")
    content_disposition = resp.headers.get("content-disposition", "")

    if not content_type in {"application/zip", "binary/octet-stream", "application/octet-stream", "multipart/form-data"}:
        if not (content_type == "unknown" and "attachment" in content_disposition):
            raise RuntimeError(f"URL has no file content, found '{content_type}' instead. Check your keys.")

    if not filename:
        if content_disposition:
            filename = re.findall("filename=\"(.+)\";?", content_disposition)[0]
        else:
            filename = url.rsplit("/",-1)[-1]
        filename = os.path.join(folder, filename)

    if started_callback:
        started_callback(filename, resp)

    desc = filename.rsplit(os.path.sep)[-1]

    last = None
    with open(filename+".tmp", 'wb') as file, tqdm.tqdm(desc=desc, total=total, unit='iB', unit_scale=True, unit_divisor=1024) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)
            if not last or time.time() - last > 0.5:
                last = time.time()
                if progress_callback:
                    progress_callback(bar.format_dict)

    os.rename(filename+".tmp", filename)

def do_download(request, folder, id, callback):
    type = request["type"]
    url = request["url"]
    token = None
    
    def started(callback, label, id):
        callback({"type":"download", "data":{"status": "started", "label": label}}, id)

    def success(callback, id, label=None):
        if label:
            callback({"type":"download", "data":{"status": "success", "label": label}}, id)
        else:
            callback({"type":"download", "data":{"status": "success"}}, id)
    
    def error(callback, err, id, trace=""):
        callback({"type":"download", "data":{"status": "error", "message": err, "trace": trace}}, id)

    def requests_download(callback, folder, url, headers, id):
        def progress_callback(progress):
            if not progress["rate"]:
                return

            value = progress["n"] / progress["total"]
            rate = (progress["rate"] or 1) / (1024*1024)
            eta = (progress["total"] - progress["n"]) / (progress["rate"] or 1)

            callback({"type":"download", "data":{"status": "progress", "progress": value, "rate": rate, "eta": eta}}, id)

        def started_callback(path, response):
            filename = path.rsplit(os.path.sep)[-1]
            started(callback, filename, id)
        
        try:
            _download(url, folder, progress_callback, started_callback, headers)
            success(callback, id)
        except Exception as e:
            trace = log_traceback("DOWNLOAD")
            error(callback, str(e), id, trace)

    folder = os.path.join(folder, type)

    if not os.path.exists(folder):
        os.mkdir(folder)
    
    if 'civitai.com' in url or 'civitai.red' in url:
        if not "/api/download/models" in url:
            error(callback, "Unsupported URL", id)
            return
        token = request.get("civitai_token", None)

    if 'huggingface' in url:
        url = url.replace("/blob/", "/resolve/")
        token = request.get("hf_token", None)

    headers = {}
    if token:
        print(token)
        headers = {"Authorization": f"Bearer {token}"}
    
    thread = threading.Thread(target=requests_download, args=([callback, folder, url, headers, id]))
    thread.start()

class Inference(threading.Thread):
    def __init__(self, wrapper, callback):
        super().__init__(daemon=True)
        
        self.wrapper = wrapper
        wrapper.callback = self.got_response

        self.callback = callback
        self.requests = queue.Queue()
        self.current = None

        self.stay_alive = True

    def got_response(self, response, id=None):
        if id == None:
            id = self.current
        return self.callback(id, response)

    def run(self):
        while self.stay_alive:
            try:
                client, self.current, request = self.requests.get(False)
                convert_all_paths(request)

                if request["type"] == "txt2img":
                    #print(request["data"])
                    self.wrapper.reset()
                    self.wrapper.set(**request["data"])
                    self.wrapper.txt2img()
                elif request["type"] == "img2img":
                    #print({k:v for k,v in request["data"].items() if k not in {"image", "mask"}})
                    self.wrapper.reset()
                    self.wrapper.set(**request["data"])
                    self.wrapper.img2img()
                elif request["type"] == "upscale":
                    #print({k:v for k,v in request["data"].items() if k not in {"image", "mask"}})
                    self.wrapper.set(**request["data"])
                    self.wrapper.upscale()
                elif request["type"] == "options":
                    self.wrapper.reset()
                    self.wrapper.options()
                elif request["type"] == "manage":
                    self.wrapper.reset()
                    self.wrapper.set(**request["data"])
                    self.wrapper.manage()
                elif request["type"] == "metadata":
                    self.wrapper.reset()
                    self.wrapper.set(**request["data"])
                    self.wrapper.metadata()
                elif request["type"] == "download":
                    do_download(request["data"], self.wrapper.model_path, self.current, self.got_response)
                elif request["type"] == "chunk":
                    self.upload(**request["data"])
                elif request["type"] == "ping":
                    self.got_response({"type":"pong"})

                self.requests.task_done()
            except queue.Empty:
                time.sleep(0.01)
            except Exception as e:
                self.requests.task_done()
                if str(e) == "Read-only":
                    self.got_response({"type":"error", "data":{"message": "Server is read-only"}})
                elif str(e) == "Aborted":
                    self.got_response({"type":"aborted", "data":{}})
                else:
                    additional = ""
                    trace = ""
                    try:
                        trace = log_traceback("SERVER")
                        s = traceback.extract_tb(e.__traceback__).format()
                        s = [e for e in s if not "venv" in e][-1]
                        s = s.split(", ")
                        file = s[0].split(os.path.sep)[-1][:-1]
                        line = s[1].split(" ")[1]
                        additional = f" ({file}:{line})"
                    except Exception as a:
                        trace = log_traceback("LOGGING")
                        additional = " THEN " + str(a)
                    self.got_response({"type":"error", "data":{"message":str(e) + additional, "trace": trace}})
    
    def upload(self, type, name, chunk=None, index=-1, total=0):
        global UPLOAD_IDS
        rel_file = os.path.join(type, name)
        file = os.path.join(self.wrapper.model_path, rel_file)
        os.makedirs(os.path.dirname(file), exist_ok=True)

        tmp = file + ".tmp"

        if index == 0:
            id = self.current
            UPLOAD_IDS[file] = id
            self.got_response({"type":"download", "data":{"status": "started", "label": rel_file.split(os.path.sep)[-1]}}, id)
            if os.path.exists(tmp):
                os.remove(tmp)

        id = UPLOAD_IDS[file]
        if chunk:
            with open(tmp, 'ab') as f:
                f.write(chunk)
            self.got_response({"type":"download", "data":{"status": "progress", "progress": index/total, "rate": 1, "eta": 0}}, id)
        else:
            if os.path.exists(tmp):
                os.rename(tmp, file)
            self.got_response({"type":"download", "data":{"status": "success"}}, id)

    def fetch(self, result_id, request_id, queue):
        def do_fetch():
            result = self.wrapper.fetch(result_id)
            if not result:
                return
            queue.put((request_id, result))

        thread = threading.Thread(target=do_fetch, args=([]), daemon=True)
        thread.start()

class Server():
    def __init__(self, wrapper, host, port, password=DEFAULT_PASSWORD):
        self.stopping = False

        self.requests = {}
        self.clients = {}
        self.reconnected = {}

        self.scheme = None
        if password != None:
            self.scheme = get_scheme(password)

        self.inference = Inference(wrapper, callback=self.on_response)
        self.server = websockets.sync.server.serve(self.handle_connection, host=host, port=int(port), max_size=None)
        self.serve = threading.Thread(target=self.serve_forever, daemon=True)

    def start(self):
        print("SERVER: starting")
        self.inference.start()
        self.serve.start()

    def stop(self):
        print("SERVER: stopping")
        self.stopping = True
        self.inference.stay_alive = False
        print("SERVER: shutdown")
        self.server.shutdown()
        print("SERVER: join")
        self.join()
        print("SERVER: done")

    def join(self, timeout=None):
        self.serve.join(timeout)
        if self.serve.is_alive():
            return False # timeout
        self.inference.join()
        return True

    def serve_forever(self):
        self.server.serve_forever()

    def terminate_connection(self, connection):
        connection.protocol.send_frame = lambda x: None
        connection.socket.close()
        raise websockets.exceptions.ConnectionClosedError(None, None)

    def handle_connection(self, connection: websockets.sync.server.ServerConnection):
        new = True

        client_id = get_id()

        self.clients[client_id] = queue.Queue()
        self.clients[client_id].put((-1, {"type":"hello", "data":{"id":client_id}}))

        lost = False
        waiting = 0
        try:
            while not self.stopping:
                if not self.clients[client_id].empty():
                    id, response = self.clients[client_id].get()
                    response["id"] = id
                    data = encrypt(self.scheme, bson_encode(response))
                    data = [data[i:min(i+FRAGMENT_SIZE,len(data))] for i in range(0, len(data), FRAGMENT_SIZE)]
                    connection.send(data)
                else:
                    data = None
                    try:
                        data = connection.recv(timeout=0)
                        waiting = 0
                    except TimeoutError:
                        pass
                    if not data:
                        time.sleep(0.01)
                        waiting += 10
                        if waiting >= 2000:
                            waiting = 0
                            pong = connection.ping()
                            #if not pong.wait(10):
                            #    self.terminate_connection(connection)

                        continue
                    error = None
                    request = None
                    if type(data) in {bytes, bytearray}:
                        try:
                            data = decrypt(self.scheme, bytes(data))
                            try:
                                request = bson_decode(data)
                            except:
                                error = "Malformed request"
                        except:
                            error = "Incorrect password"
                    else:
                        error = "Invalid request"
                    if request:
                        if request["type"] == "options" and new:
                            print(f"SERVER: client connected")
                            new = False
                        if request["type"] == "cancel":
                            id = request["data"]["id"]
                            if id in self.requests and self.requests[id] == client_id:
                                del self.requests[id]
                                self.send_response(client_id, id, {'type': 'aborted', 'data': {}})
                        if request["type"] == "reconnect":
                            old_client_id = request["data"]["id"]
                            if old_client_id in self.clients:
                                while not self.clients[old_client_id].empty():
                                    self.clients[client_id].put(self.clients[old_client_id].get())
                            self.reconnected[old_client_id] = client_id
                            for id in list(self.requests.keys()):
                                if self.requests[id] == old_client_id:
                                    self.requests[id] = client_id
                            
                        request_id = get_id()
                        if "id" in request:
                            request_id = request["id"]
                        self.requests[request_id] = client_id

                        if request["type"] == "fetch":
                            self.inference.fetch(request["data"]["id"], request_id, self.clients[client_id])
                            continue

                        remaining = self.inference.requests.unfinished_tasks
                        self.inference.requests.put((client_id, request_id, request))
                        self.clients[client_id].put((-1, {"type":"ack", "data":{"id": request_id, "queue": remaining}}))
                    else:
                        self.clients[client_id].put((-1, {"type":"error", "data":{"message": error}}))
        except websockets.exceptions.ConnectionClosedOK:
            pass
        except websockets.exceptions.ConnectionClosedError as e:
            if not e.rcvd and not e.sent:
                lost = True
        except Exception:
            log_traceback("CLIENT")

        if not new:
            if lost:
                print(f"SERVER: client lost, waiting...")
                for _ in range(60):
                    if client_id in self.reconnected:
                        print(f"SERVER: client found")
                        break
                    time.sleep(1)
            else:
                print(f"SERVER: client disconnected")

        if client_id in self.clients:
            del self.clients[client_id]

    def send_response(self, client, id, response):
        if client in self.clients:
            self.clients[client].put((id, response))

    def on_response(self, id, response):
        if self.stopping:
            return False
        if id in self.requests:
            client = self.requests[id]
            self.send_response(client, id, response)
            return client in self.clients
        else:
            return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='comfy-inference-server')
    parser.add_argument('--bind', type=str, help='address (ip:port) to listen on', default="0.0.0.0:28888")
    parser.add_argument('--password', type=str, help='password to derive encryption key from', default=DEFAULT_PASSWORD)
    parser.add_argument('--comfyui', type=str, help='path to ComfyUI directory', default="ComfyUI")

    args = parser.parse_args()

    ip, port = args.bind.rsplit(":",1)

    comfyui_path = os.path.abspath(args.comfyui)
    if comfyui_path not in sys.path:
        sys.path.insert(0, comfyui_path)
    import wrapper
    wrapper_ = wrapper.GenerationParameters(args.comfyui)

    server = Server(wrapper_, ip, port, args.password)
    server.start()
    
    try:
        server.join()
    except KeyboardInterrupt:
        server.stop()

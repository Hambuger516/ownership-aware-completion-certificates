"""Real loopback control messages with an independent byte-page worker.

Run on the target host too. This validates same-machine transport/accounting,
not CUDA execution or cross-machine failure recovery.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import socket
import socketserver
import threading
import time

from .runtime import AllocationAuthority
from .model import State, Group


class BytePageWorker:
    def __init__(self, capacity=4, page_bytes=4096):
        self.capacity = capacity
        self.page_bytes = page_bytes
        self.storage = {}
        self.allocations = 0

    def snapshot(self):
        if not self.storage:
            return State((self.capacity,),(),()), ()
        pages = len(self.storage)
        groups = tuple(Group(str(k),0,1,frozenset({0})) for k in self.storage)
        return State((self.capacity-pages,),((3-pages,),),groups), ("r",)

    def allocate(self):
        # Storage is genuinely materialized and touched before confirmation.
        if self.storage:
            raise RuntimeError("duplicate physical dispatch")
        self.storage[1] = bytearray(self.page_bytes)
        self.storage[1][:4] = b"PRC!"
        self.allocations += 1
        return [1]

    def release(self):
        self.storage.clear()


class ProtocolProbe:
    def __init__(self):
        self.events = []
        self.worker = BytePageWorker()
        self.authority = AllocationAuthority(self.worker.snapshot()[0], on_event=self.events.append)
        self.grants = {}
        self.delayed = threading.Event()

    def handle(self, message):
        if message.get("delay_ms"):
            self.delayed.set()
            time.sleep(message["delay_ms"]/1000)
        op = message["op"]
        if op == "authorize":
            post = State((3,),((2,),),(Group("planned",0,1,frozenset({0})),))
            grant = self.authority.authorize(message["id"],"r",post,("r",),expected_version=message["version"])
            self.grants[message["id"]] = grant
            return {"grant_id":grant.grant_id,"virtual_free":self.authority.virtual.free}
        if op == "execute":
            result = self.authority.dispatch(self.grants[message["id"]],self.worker.allocate,self.worker.snapshot)
            return {"pages":result}
        if op == "stop":
            self.authority.stop_unexecuted(self.grants[message["id"]],self.worker.snapshot)
            return {"stopped":True}
        if op == "release":
            return {"outcome":self.authority.release("r",self.worker.release,self.worker.snapshot,cancelled=True)}
        if op == "state":
            return {"version":self.authority.version,"confirmed_free":self.authority.confirmed.free,
                    "virtual_free":self.authority.virtual.free,"allocations":self.worker.allocations}
        raise ValueError("unknown control operation")


def run_probe():
    probe = ProtocolProbe()
    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            try:
                raw = self.rfile.readline(65537)
                if len(raw)>65536:
                    raise ValueError("message too large")
                result = {"ok":True,"result":probe.handle(json.loads(raw))}
            except Exception as exc:
                result = {"ok":False,"error_type":type(exc).__name__,"error":str(exc)}
            self.wfile.write((json.dumps(result)+"\n").encode())
    class Server(socketserver.ThreadingTCPServer):
        daemon_threads = True
        allow_reuse_address = True
    with Server(("127.0.0.1",0),Handler) as server:
        thread = threading.Thread(target=server.serve_forever,kwargs={"poll_interval":0.02},daemon=True)
        thread.start()
        history = []
        def rpc(**message):
            with socket.create_connection(server.server_address,timeout=3) as connection:
                connection.sendall((json.dumps(message)+"\n").encode())
                response = json.loads(connection.makefile("rb").readline())
            history.append(dict(message=message,response=response))
            return response
        try:
            issued = rpc(op="authorize",id="a0",version=0)
            duplicate = rpc(op="authorize",id="a0",version=0)
            assert issued == duplicate and issued["ok"]
            with ThreadPoolExecutor(max_workers=1) as clients:
                delayed = clients.submit(rpc,op="execute",id="a0",delay_ms=100)
                assert probe.delayed.wait(2)
                # Delay alone has NOT refunded the pending right.
                pending = rpc(op="state")["result"]
                assert pending["virtual_free"] == [3] and pending["confirmed_free"] == [4]
                assert rpc(op="stop",id="a0")["ok"]
                late = delayed.result(timeout=3)
                assert not late["ok"] and late["error_type"] == "Refused"
            assert probe.worker.allocations == 0
            old = rpc(op="authorize",id="old",version=0)
            assert not old["ok"] and old["error_type"] == "StaleState"
            version = rpc(op="state")["result"]["version"]
            assert rpc(op="authorize",id="a1",version=version)["ok"]
            executed = rpc(op="execute",id="a1")
            assert executed == rpc(op="execute",id="a1") and executed["ok"]
            assert probe.worker.allocations == 1 and probe.worker.storage[1][:4] == b"PRC!"
            assert rpc(op="release")["ok"]
            assert not probe.worker.storage
            return {"status":"PASS","transport":"TCP loopback with concurrent handlers",
                    "backend":"independent materialized CPU byte pages; not CUDA",
                    "messages":history,"authority_events":probe.events,
                    "physical_allocations":probe.worker.allocations,
                    "final_physical_free":probe.worker.snapshot()[0].free,
                    "coverage":["duplicate authorize","delay without refund","stop before late execution",
                                "stale version","duplicate execution","confirmed physical cancellation release"]}
        finally:
            server.shutdown()
            thread.join(timeout=3)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=run_probe()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,indent=2)
    print(json.dumps({"status":result["status"],"messages":len(result["messages"]),"backend":result["backend"]}))


if __name__=="__main__":
    main()

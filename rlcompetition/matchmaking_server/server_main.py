from concurrent import futures
import time
import logging

import grpc

import zmq

from threading import Thread


from grpc_gen.server_pb2 import QuickMatchReply
from grpc_gen.server_pb2_grpc import MatchmakerServicer, add_MatchmakerServicer_to_server

_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class MatchMakingHandler(MatchmakerServicer):

    def GetMatch(self, request, context):
        #  Prepare our context and sockets
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect("ipc://matchmaker_requests")

        socket.send_pyobj(request)
        return socket.recv_pyobj()

class MatchProcessJanitor(Thread):

    def run(self) -> None:




class MatchmakingThread(Thread):

    def __init__(self):
        super().__init__()

        # Prepare our context and sockets
        context = zmq.Context()
        self.socket = context.socket(zmq.REP)
        self.socket.bind("ipc://matchmaker_requests")
        print("Matchmaker thread listening...")

    def run(self) -> None:
        while True:

            request = self.socket.recv_pyobj()
            print("Got request from {}".format(request.username))
            response = QuickMatchReply(server='localhost:7676', auth_key="kjghdekjghrkegh")
            self.socket.send_pyobj(response)


def serve():

    matchmaker_thread = matchmaker_service()
    matchmaker_thread.start()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_MatchmakerServicer_to_server(MatchMakingHandler(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    logging.basicConfig()
    serve()
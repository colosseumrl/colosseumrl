from concurrent import futures
import time
import logging

import grpc

import zmq

from threading import Thread


from .grpc_gen.server_pb2 import QuickMatchReply, QuickMatchRequest
from .grpc_gen.server_pb2_grpc import MatchmakerServicer, add_MatchmakerServicer_to_server

_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class MatchMakingHandler(MatchmakerServicer):

    def GetMatch(self, request, context):
        #  Prepare our context and sockets
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect("ipc://matchmaker_requests")
        # print(request.SerializeToString(), QuickMatchRequest.FromString(request.SerializeToString()))
        socket.send(request.SerializeToString())
        return QuickMatchReply.FromString(socket.recv())

class MatchProcessJanitor(Thread):

    def run(self) -> None:
        pass


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

            request = QuickMatchRequest.FromString(self.socket.recv())
            print("Got request from {}".format(request.username))
            response = QuickMatchReply(server='localhost:7676', auth_key="kjghdekjghrkegh")
            self.socket.send(response.SerializeToString())


def serve():

    matchmaker_thread = MatchmakingThread()
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
from __future__ import print_function
import logging

import grpc

from .grpc_gen.server_pb2 import QuickMatchRequest
from .grpc_gen.server_pb2_grpc import MatchmakerStub

import random

def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = MatchmakerStub(channel)
        username = 'noobmaster{}'.format(random.randint(1,100))
        print("Acting as {}".format(username))
        response = stub.GetMatch(QuickMatchRequest(username=username))
    print("Got Match: " + response.server + " " + response.auth_key + " " + response.username)


if __name__ == '__main__':
    logging.basicConfig()
    run()
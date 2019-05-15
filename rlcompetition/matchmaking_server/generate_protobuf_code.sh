#!/usr/bin/env bash

python -m grpc_tools.protoc -I . --python_out=. --grpc_python_out=. grpc_gen/server.proto

sed -i 's/from grpc_gen import/from . import/g' grpc_gen/server_pb2_grpc.py

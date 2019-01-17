from Environment import BaseEnvironment
from Datamodel import Player
from rtypes import pcc_set, dimension, primarykey
from spacetime import Application
import spacetime

from argparse import ArgumentParser


def server_manager(dataframe, game):
    pass


def main():
    server = Application(server_manager, server_port=7777,
                         Types=[Player], version_by=spacetime.utils.enums.VersionBy.FULLSTATE)

if __name__ == '__main__':
    main()
'''
@package tfignite.parser
Customized parser for tfignite.
'''
import argparse


class ArgumentParser(argparse.ArgumentParser):
    def __init__(self, Model, Dataset, *args, **kwargs):
        self._Model = Model
        self._Dataset = Dataset
        argparse.ArgumentParser.__init__(self, *args, **kwargs)

    def parse_args(self, *args, **kwargs):
        model_parser = self.add_argument_group(self._Model.__name__)
        self._Model.add_parser(model_parser)

        dataset_parser = self.add_argument_group(self._Dataset.__name__)
        self._Dataset.add_parser(dataset_parser)

        return argparse.ArgumentParser.parse_args(self, *args, **kwargs)

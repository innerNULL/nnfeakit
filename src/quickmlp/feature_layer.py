# -*- coding: utf-8 -*-
# file: feature_layer.py
# date: 2023-09-28


import pdb
import torch
from enum import Enum
from typing import Union, List, Optional, Set, Dict
from torch import device
from torch import Tensor, LongTensor, FloatTensor
from torch.nn import Module, ModuleDict
from torch.nn import Linear, Softmax, Parameter, Embedding


FEATURE_TYPES: Set[str] = {"float", "int", "array"}


class FeatureColumn(Module):
    def __init__(self, 
        fea_name: str, fea_type: str, fea_space_size: int=-1, 
        in_dim: int=1, out_dim: int=1, padding_idx: int=-1
    ):
        super().__init__()
        self.name: str = ""
        self.type: str = None
        self.space_size: Optional[int] = None
        self.in_dim: int = 1
        self.out_dim: int = 1

        self.name = fea_name

        if fea_type in FEATURE_TYPES:
            self.type = fea_type
        else:
            raise "Illegal `fea_type` value."

        if fea_type == "int":
            if fea_space_size <= 0:
                raise "Illegal `fea_space_size` value."
            else:
                self.space_size = fea_space_size

        if fea_type == "int":
            self.in_dim = in_dim
            self.out_dim = out_dim
        if fea_type == "array":
            self.in_dim = in_dim
            self.out_dim = in_dim

    def _forward_check(self, feature: Tensor) -> None:
        assert(len(feature.shape) == 2)
        assert(feature.shape[1] == self.in_dim)


class FloatFeatureColumn(FeatureColumn):
    def __init__(self,
        fea_name: str, fea_type: str, fea_space_size: int=-1,
        in_dim: int=1, out_dim: int=1, padding_idx: int=-1
    ):
        super().__init__(fea_name, fea_type, fea_space_size, in_dim, out_dim)
        assert(self.type == "float")

    def forward(self, feature: FloatTensor) -> FloatTensor:
        self._forward_check(feature)
        return feature

    @classmethod
    def new(cls, conf: Dict[str, Union[int, str]]):
        return cls(
            fea_name=conf["name"], fea_type="float",
            fea_space_size=-1, in_dim=1, out_dim=1, padding_idx=-1
        )


class IntFeatureColumn(FeatureColumn):
    def __init__(self, 
        fea_name: str, fea_type: str, fea_space_size: int=-1, 
        in_dim: int=1, out_dim: int=1, padding_idx: int=-1, 
        merger: str="null"
    ):
        super().__init__(fea_name, fea_type, fea_space_size, in_dim, out_dim)
        assert(self.type == "int")
        assert(out_dim < fea_space_size and out_dim > 1)
        assert(padding_idx >= 0)
        assert(merger in {"sum", "mean", "concat"})
        
        self.embedding_dim: int = out_dim
        self.merger: str = merger

        if merger == "concat":
            self.out_dim = self.out_dim * self.in_dim

        self.embedding: Embedding = Embedding(
            num_embeddings=self.space_size, embedding_dim=self.embedding_dim, 
            padding_idx=padding_idx
        )

    def forward(self, feature: LongTensor) -> FloatTensor:
        self._forward_check(feature)
        if self.merger == "sum":
            return self.embedding(feature).sum(dim=-2)
        elif self.merger == "mean":
            return self.embedding(feature).mean(dim=-2)
        elif self.merger == "concat":
            return self.embedding(feature).view(-1, self.out_dim)

    @classmethod
    def new(cls, conf: Dict[str, Union[int, str]]):
        return cls(
            fea_name=conf["name"], fea_type="int", 
            fea_space_size=conf["fea_space_size"], 
            in_dim=conf["in_dim"], out_dim=conf["out_dim"], 
            padding_idx=conf["padding_idx"], 
            merger=conf["merger"]
        )


class ArrayFeatureColumn(FeatureColumn):
    def __init__(self,
        fea_name: str, fea_type: str, fea_space_size: int=-1,
        in_dim: int=1, out_dim: int=1, padding_idx: int=-1
    ):
        super().__init__(fea_name, fea_type, fea_space_size, in_dim, out_dim)
        assert(self.type == "array")
        assert(in_dim == out_dim)

    def forward(self, feature: FloatTensor) -> FloatTensor:
        self._forward_check(feature)
        return feature

    @classmethod
    def new(cls, conf: Dict[str, Union[int, str]]):
        return cls(
            fea_name=conf["name"], fea_type="array",
            fea_space_size=-1, in_dim=conf["in_dim"], out_dim=conf["in_dim"], 
            padding_idx=-1
        )


class FeatureLayer(Module):
    def __init__(self, features: List[Dict[str, Union[int, str]]]):
        super().__init__()
        self.feature_names: List[str] = []
        self.feature_columns: ModuleDict = ModuleDict()
        self.out_dim: int = 0
        
        for fea_conf in features:
            fea_name: str = fea_conf["name"]
            fea_type: str = fea_conf["type"]

            if fea_name in self.feature_columns:
                raise "Feature '%s' already exists." % fea_name
            
            fea_col: FeatureColumn = None
            if fea_type == "int":
                fea_col = IntFeatureColumn.new(fea_conf)
            elif fea_type == "float":
                fea_col = FloatFeatureColumn.new(fea_conf)
            elif fea_type == "array":
                fea_col = ArrayFeatureColumn.new(fea_conf)
            else:
                raise "Illegal feature type '%s'" % fea_type

            self.feature_names.append(fea_name)
            self.feature_columns[fea_name] = fea_col
            self.out_dim += fea_col.out_dim

    def forward(self, inputs: Dict[str, Tensor]) -> FloatTensor:
        for feature_name in self.feature_names:
            assert(feature_name in inputs)

        feature_vals: List[FloatTensor] = []
        for feature_name in self.feature_names:
            feature_inputs: Tensor = inputs[feature_name]
            feature_vals.append(
                self.feature_columns[feature_name](feature_inputs)
            )
        return torch.cat(feature_vals, dim=-1)


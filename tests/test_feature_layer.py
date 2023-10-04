# -*- coding: utf-8 -*-
# file: feature_layer.py
# date: 2023-09-28


import sys, os
sys.path.append(
    os.path.join(os.path.dirname(__file__), "../src")
)

import pdb
import random 
from typing import List, Dict
from torch import Tensor, LongTensor, FloatTensor

from quickmlp.feature_layer import IntFeatureColumn, FloatFeatureColumn, ArrayFeatureColumn
from quickmlp.feature_layer import FeatureLayer


def test_FloatFeatureColumn_0() -> None:
    error: int = 0
    fea_col: FloatFeatureColumn = None

    error = 0
    try:
        fea_col = FloatFeatureColumn(
            fea_name="fea1", fea_type="array",
            fea_space_size=-1, in_dim=-1, out_dim=-1
        )
    except Exception as e:
        error = 1
    assert(error == 1)

    error = 0
    try:
        fea_col = FloatFeatureColumn(
            fea_name="fea1", fea_type="float", 
            fea_space_size=-1, in_dim=-1, out_dim=-1
        )
    except Exception as e:
        assert(error == 0)


def test_FloatFeatureColumn_1() -> None:
    fea_col: FloatFeatureColumn = FloatFeatureColumn(
        fea_name="fea1", fea_type="float", fea_space_size=-1, in_dim=1, out_dim=1
    )

    # Single value 
    in_val: FloatTensor = FloatTensor([[3.24]])
    out_val: FloatTensor = fea_col(in_val)
    assert(in_val == out_val)

    # A batch of value
    batch_size: int = 8
    batch_in_val: FloatTensor = FloatTensor([[3.24]] * 8)
    batch_out_val: FloatTensor = fea_col(batch_in_val)
    assert(batch_in_val.shape == batch_out_val.shape)
    for i in range(batch_size):
        assert(batch_in_val[i] == batch_out_val[i])


def test_IntFeatureColumn_0() -> None:
    error: int = 0
    fea_col: IntFeatureColumn = None

    error = 0
    try:
        fea_col = IntFeatureColumn(
            fea_name="fea1", fea_type="float", 
            fea_space_size=128, in_dim=10, out_dim=16, merger="mean"
        )
    except Exception as e:
        error = 1
    assert(error == 1)

    error = 0
    try:
        fea_col = IntFeatureColumn(
            fea_name="fea1", fea_type="int",
            fea_space_size=128, in_dim=20, out_dim=256, merger="mean"
        )
    except Exception as e:
        error = 1
    assert(error == 1)

    error = 0
    try:
        fea_col = IntFeatureColumn(
            fea_name="fea1", fea_type="int",
            fea_space_size=10000, in_dim=15, out_dim=256, merger="mean"
        )
    except Exception as e:
        error = 1
    assert(error == 1)

    error = 0
    try:
        fea_col = IntFeatureColumn(
            fea_name="fea1", fea_type="int",
            fea_space_size=10000, in_dim=15, out_dim=1, 
            padding_idx=0, merger="mean"
        )
    except Exception as e:
        error = 1
    assert(error == 1)

    error = 0
    try:
        fea_col = IntFeatureColumn(
            fea_name="fea1", fea_type="int",
            fea_space_size=10000, in_dim=15, out_dim=256, 
            padding_idx=0, merger="mean"
        )
    except Exception as e:
        error = 1
    assert(error == 0)


def test_IntFeatureColumn_1() -> None:
    embedding_dim: int = 256
    in_dim: int = 5
    fea_col: IntFeatureColumn = IntFeatureColumn(
        fea_name="fea1", fea_type="int", 
        fea_space_size=10000, in_dim=5, out_dim=embedding_dim, 
        padding_idx=0, merger="mean"
    )
    single_input: LongTensor = LongTensor([[1, 22, 33, 444, 5555]])
    single_output: FloatTensor = fea_col(single_input)
    assert(single_input.shape[-1] == in_dim)
    assert(single_output.shape[-1] == embedding_dim)


def test_IntFeatureColumn_2() -> None:
    embedding_dim: int = 256
    in_dim: int = 5
    fea_col: IntFeatureColumn = IntFeatureColumn(
        fea_name="fea1", fea_type="int", 
        fea_space_size=10000, in_dim=5, out_dim=embedding_dim, 
        padding_idx=0, merger="concat"
    )
    single_input: LongTensor = LongTensor([[1, 22, 33, 444, 0]])
    single_output: FloatTensor = fea_col(single_input)
    assert(single_input.shape[-1] == in_dim)
    assert(single_output.shape[-1] == embedding_dim * in_dim)
    for i in range(embedding_dim):
        assert(int(single_output[0, -(i + 1)]) == 0)


def test_ArrayFeatureColumn_0() -> None:
    error: int = 0
    fea_col: ArrayFeatureColumn = None

    error = 0
    try:
        fea_col = ArrayFeatureColumn(
            fea_name="fea1", fea_type="float", 
            fea_space_size=-1, in_dim=32, out_dim=32
        )
    except Exception as e:
        error = 1
    assert(error == 1)

    error = 0
    try:
        fea_col = ArrayFeatureColumn(
            fea_name="fea1", fea_type="array",
            fea_space_size=-1, in_dim=32, out_dim=64
        )
    except Exception as e:
        error = 1
    assert(error == 1)

    error = 0
    try:
        fea_col = ArrayFeatureColumn(
            fea_name="fea1", fea_type="array",
            fea_space_size=-1, in_dim=32, out_dim=32
        )
    except Exception as e:
        error = 1
    assert(error == 0)


def test_ArrayFeatureColumn_1() -> None:
    array_dim: int = 128
    fea_col: ArrayFeatureColumn = ArrayFeatureColumn(
        fea_name="fea1", fea_type="array",
        fea_space_size=-1, in_dim=array_dim, out_dim=array_dim
    )

    single_input: FloatTensor = FloatTensor(
        [[random.random() for i in range(array_dim)]]
    )
    single_output: FloatTensor = fea_col(single_input)
    assert(single_input.shape == single_output.shape)
    for i in range(array_dim):
        assert(single_input[0][i] == single_output[0][i])


def test_FeatureLayer_0() -> None:
    fea_layer: FeatureLayer = FeatureLayer([])


def test_FeatureLayer_1() -> None:
    error: int = 0
    fea_configs: List[Dict] = [
        {"name": "fea1", "type": "float"},
        {"name": "fea1", "type": "int", "fea_space_size": 1000, "out_dim": 128}
    ]
    
    try:
        fea_layer: FeatureLayer = FeatureLayer(fea_configs)
    except Exception as e:
        error = 1
    assert(error == 1)


def test_FeatureLayer_2() -> None:
    error: int = 0
    fea_configs: List[Dict] = [
        {"name": "fea1", "type": "float"},
        {"name": "fea2", "type": "float"}
    ]
    single_inputs: Dict[str, Tensor] = {"fea2": FloatTensor([[3.14]])}
    fea_layer: FeatureLayer = FeatureLayer(fea_configs)

    try:
        fea_layer(single_inputs)
    except Exception as e:
        error = 1
    assert(error == 1)


def test_FeatureLayer_3() -> None:
    error: int = 0
    fea_configs: List[Dict] = [
        {"name": "fea1", "type": "float"},
        {"name": "fea2", "type": "wrong_type"}
    ]

    try:
        fea_layer: FeatureLayer = FeatureLayer(fea_configs)
    except Exception as e:
        error = 1
    assert(error == 1)


def test_FeatureLayer_4() -> None:
    fea_configs: List[Dict] = [
        {"name": "fea1", "type": "float"},
        {"name": "fea2", "type": "int", "fea_space_size": 10000, "in_dim": 4, "out_dim": 128, "padding_idx": 0, "merger": "mean"}, 
        {"name": "fea3", "type": "float"}, 
        {"name": "fea4", "type": "array", "in_dim": 64}, 
        {"name": "fea5", "type": "int", "fea_space_size": 7531, "in_dim": 5, "out_dim": 256, "padding_idx": 0, "merger": "mean"},
    ]
    fake_single_inputs: Dict[Tensor] = {
        "fea1": FloatTensor([[3.14]]), 
        "fea2": LongTensor([[5, 66, 777, 8888]]), 
        "fea3": FloatTensor([[0.15926]]), 
        "fea4": FloatTensor([[random.random() for i in range(64)]]), 
        "fea5": LongTensor([[8, 77, 666, 5555, 0]])
    }
    
    fea_layer: FeatureLayer = FeatureLayer(fea_configs)
    
    single_outputs: FloatTensor = fea_layer(fake_single_inputs)
    assert(single_outputs.shape[-1] == (1 + 128 + 1 + 64 + 256))
    assert(fea_layer.out_dim == single_outputs.shape[-1])

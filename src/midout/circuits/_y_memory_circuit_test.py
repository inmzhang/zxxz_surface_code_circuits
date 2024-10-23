import itertools

import pytest
import stim

from midout import gen
from midout._make_circuit import make_circuit


@pytest.mark.parametrize("distance", range(2, 10))
def test_make_y_memory_experiment(distance: int):
    circuit = make_circuit(
        basis='Y',
        distance=distance,
        noise=gen.NoiseModel.uniform_depolarizing(1e-3),
        boundary_rounds=distance // 2,
        memory_rounds=5,
        verify_chunks=True,
    )

    dem = circuit.detector_error_model(decompose_errors=True, block_decomposition_from_introducing_remnant_edges=True)
    assert dem is not None

    actual_distance = len(circuit.shortest_graphlike_error())
    expected_distance = distance
    assert actual_distance == expected_distance

    # actual_distance2 = len(circuit.search_for_undetectable_logical_errors(
    #     dont_explore_edges_with_degree_above=4,
    #     dont_explore_edges_increasing_symptom_degree=False,
    #     dont_explore_detection_event_sets_with_size_above=4,
    #     canonicalize_circuit_errors=True,
    # ))
    # assert actual_distance2 == expected_distance


@pytest.mark.parametrize("basis,distance,boundary_rounds", itertools.product(
    ["Y"],
    [3, 4, 5],
    [0, 5]
))
def test_have_all_detectors(basis: str, distance: int, boundary_rounds: int):
    c = make_circuit(
        basis=basis,
        distance=distance,
        noise=gen.NoiseModel.uniform_depolarizing(1e-3),
        memory_rounds=3,
        boundary_rounds=boundary_rounds,
    )
    c.detector_error_model()
    gen.verify_circuit_has_all_possible_detectors(c)


def test_exact_circuit_y():
    assert make_circuit(
        basis='Y',
        distance=3,
        noise=None,
        boundary_rounds=10,
        memory_rounds=100,
    ) == stim.Circuit("""
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(0, 1) 1
        QUBIT_COORDS(0, 2) 2
        QUBIT_COORDS(1, 0) 3
        QUBIT_COORDS(1, 1) 4
        QUBIT_COORDS(1, 2) 5
        QUBIT_COORDS(2, 0) 6
        QUBIT_COORDS(2, 1) 7
        QUBIT_COORDS(2, 2) 8
        QUBIT_COORDS(-0.5, 1.5) 9
        QUBIT_COORDS(0.5, -0.5) 10
        QUBIT_COORDS(0.5, 0.5) 11
        QUBIT_COORDS(0.5, 1.5) 12
        QUBIT_COORDS(1.5, -0.5) 13
        QUBIT_COORDS(1.5, 0.5) 14
        QUBIT_COORDS(1.5, 1.5) 15
        QUBIT_COORDS(1.5, 2.5) 16
        QUBIT_COORDS(2.5, 0.5) 17
        QUBIT_COORDS(2.5, 1.5) 18
        R 15 13 11 9 6 4 3 2 1 8 7 5 18 16 14 12
        TICK
        H 2 4 5 9 11 12 13 14 15 16 18
        TICK
        CZ 1 11 2 12 3 13 4 14 5 15 8 18
        TICK
        H 1 2 3 4 5 8
        TICK
        CZ 4 15 5 12 7 14
        TICK
        CZ 1 12 2 9 3 14 4 11 5 16 6 13 7 18 8 15
        TICK
        H 1 3 4 5 6 7 8 13 18
        TICK
        CZ 1 9 3 11 4 12 6 14 7 15 8 16
        TICK
        H 4 6 8 9 11 12 14 15 16
        TICK
        M 15 13 11 9 18 16 14 12
        DETECTOR(-0.5, 1.5, 0) rec[-5]
        DETECTOR(0.5, 0.5, 0) rec[-6]
        DETECTOR(1.5, -0.5, 0) rec[-7]
        DETECTOR(1.5, 2.5, 0) rec[-3]
        DETECTOR(2.5, 1.5, 0) rec[-4]
        SHIFT_COORDS(0, 0, 1)
        TICK
        REPEAT 10 {
            R 15 13 11 9 18 16 14 12
            TICK
            H 2 4 8 9 11 12 13 14 15 16 18
            TICK
            CZ 1 11 2 12 3 13 4 14 5 15 8 18
            TICK
            H 1 2 3 4 5 7 8
            TICK
            CZ 4 15 5 12 7 14
            TICK
            CZ 1 12 2 9 3 14 4 11 5 16 6 13 7 18 8 15
            TICK
            H 1 3 4 5 6 7 8 13 18
            TICK
            CZ 1 9 3 11 4 12 6 14 7 15 8 16
            TICK
            H 4 6 8 9 11 12 14 15 16
            TICK
            M 15 13 11 9 18 16 14 12
            DETECTOR(-0.5, 1.5, 0) rec[-13] rec[-5]
            DETECTOR(0.5, 0.5, 0) rec[-14] rec[-6]
            DETECTOR(0.5, 1.5, 0) rec[-9] rec[-1]
            DETECTOR(1.5, -0.5, 0) rec[-15] rec[-7]
            DETECTOR(1.5, 0.5, 0) rec[-10] rec[-2]
            DETECTOR(1.5, 1.5, 0) rec[-16] rec[-8]
            DETECTOR(1.5, 2.5, 0) rec[-11] rec[-3]
            DETECTOR(2.5, 1.5, 0) rec[-12] rec[-4]
            SHIFT_COORDS(0, 0, 1)
            TICK
        }
        R 15 13 11 10 9 0 18 17 16 14 12
        TICK
        C_ZYX 11 15
        H 4 8 9 10 12 13 16
        SQRT_X 0 14 18
        TICK
        CZ 4 14 8 18
        TICK
        CZ 1 9 3 13 4 12 7 17 8 16
        TICK
        C_XYZ 4 8
        C_ZYX 14 18
        H 1 3 5 7 17
        TICK
        CZ 1 12 2 9 3 14 4 11 5 16 6 13 7 18 8 15
        TICK
        C_ZYX 8
        H 1 2 3 6 7 9 13 14 16 18
        SQRT_X 4
        TICK
        CZ 0 11 3 10 4 15 5 12 6 17 7 14
        TICK
        H 0 3 4 5 6 7
        TICK
        CZ 0 10 1 11 2 12 4 14 5 15 7 17
        TICK
        H 0 2 5 10 11 12 14 15 17
        TICK
        M 18 17 15 11 9 16 14 13 12 10
        DETECTOR(-0.5, 1.5, 0) rec[-15] rec[-6]
        DETECTOR(0.5, 0.5, 0) rec[-16] rec[-7] rec[-1]
        DETECTOR(0.5, 1.5, 0) rec[-11] rec[-2]
        DETECTOR(1.5, -0.5, 0) rec[-17] rec[-3]
        DETECTOR(1.5, 0.5, 0) rec[-12] rec[-9]
        DETECTOR(1.5, 1.5, 0) rec[-18] rec[-8] rec[-4]
        DETECTOR(1.5, 2.5, 0) rec[-13] rec[-5]
        DETECTOR(2.5, 1.5, 0) rec[-14] rec[-10]
        OBSERVABLE_INCLUDE(0) rec[-10] rec[-9] rec[-5] rec[-2]
        SHIFT_COORDS(0, 0, 1)
        TICK
        R 10 12 14 16 9 11 15 17
        TICK
        H 9 10 11 12 14 15 16 17
        TICK
        CZ 1 9 3 11 4 12 6 14 7 15 8 16
        TICK
        H 1 3 4 6 7 8
        TICK
        CZ 1 12 2 9 3 14 4 11 5 16 8 15
        TICK
        CZ 0 11 3 10 4 15 5 12 6 17 7 14
        TICK
        H 0 1 2 3 4 5 7 9
        TICK
        CZ 0 10 1 11 2 12 4 14 5 15 7 17
        TICK
        H 0 2 4 10 11 12 14 15 16 17
        TICK
        M 10 12 14 16 9 11 15 17
        DETECTOR(-0.5, 1.5, 0) rec[-14] rec[-4]
        DETECTOR(0.5, -0.5, 0) rec[-9] rec[-8]
        DETECTOR(0.5, 0.5, 0) rec[-15] rec[-3]
        DETECTOR(0.5, 1.5, 0) rec[-10] rec[-7]
        DETECTOR(1.5, 0.5, 0) rec[-12] rec[-11] rec[-6]
        DETECTOR(1.5, 1.5, 0) rec[-18] rec[-16] rec[-2]
        DETECTOR(1.5, 2.5, 0) rec[-13] rec[-5]
        DETECTOR(2.5, 0.5, 0) rec[-17] rec[-1]
        SHIFT_COORDS(0, 0, 1)
        TICK
        REPEAT 99 {
            R 10 12 14 16 9 11 15 17
            TICK
            H 4 6 8 9 10 11 12 14 15 16 17
            TICK
            CZ 1 9 3 11 4 12 6 14 7 15 8 16
            TICK
            H 1 3 4 5 6 7 8
            TICK
            CZ 1 12 2 9 3 14 4 11 5 16 8 15
            TICK
            CZ 0 11 3 10 4 15 5 12 6 17 7 14
            TICK
            H 0 1 2 3 4 5 7 9
            TICK
            CZ 0 10 1 11 2 12 4 14 5 15 7 17
            TICK
            H 0 2 4 10 11 12 14 15 16 17
            TICK
            M 10 12 14 16 9 11 15 17
            DETECTOR(-0.5, 1.5, 0) rec[-12] rec[-4]
            DETECTOR(0.5, -0.5, 0) rec[-16] rec[-8]
            DETECTOR(0.5, 0.5, 0) rec[-11] rec[-3]
            DETECTOR(0.5, 1.5, 0) rec[-15] rec[-7]
            DETECTOR(1.5, 0.5, 0) rec[-14] rec[-6]
            DETECTOR(1.5, 1.5, 0) rec[-10] rec[-2]
            DETECTOR(1.5, 2.5, 0) rec[-13] rec[-5]
            DETECTOR(2.5, 0.5, 0) rec[-9] rec[-1]
            SHIFT_COORDS(0, 0, 1)
            TICK
        }
        R 10 12 13 14 16 9 11 15 17 18
        TICK
        H 0 2 4 9 10 11 12 13 14 15 16 17 18
        TICK
        CZ 0 10 1 11 2 12 4 14 5 15 7 17
        TICK
        H 0 1 2 3 4 5 7
        TICK
        CZ 0 11 3 10 4 15 5 12 6 17 7 14
        TICK
        H 3 6 7 10 14 17
        SQRT_X 0 4 8
        TICK
        CZ 1 12 2 9 3 14 4 11 5 16 6 13 7 18 8 15
        TICK
        C_XYZ 11 14 15 18
        C_ZYX 4 8
        H 1 3 6 7
        TICK
        CZ 1 9 3 13 4 12 7 17 8 16
        TICK
        CZ 4 14 8 18
        TICK
        H 9 12 13 16
        SQRT_X 14 18
        TICK
        M 12 14 16 17 18 0 9 10 11 13 15
        DETECTOR(-0.5, 1.5, 0) rec[-15] rec[-5]
        DETECTOR(0.5, -0.5, 0) rec[-19] rec[-4]
        DETECTOR(0.5, 0.5, 0) rec[-14] rec[-10] rec[-3]
        DETECTOR(0.5, 1.5, 0) rec[-18] rec[-11]
        DETECTOR(1.5, 0.5, 0) rec[-17] rec[-2]
        DETECTOR(1.5, 1.5, 0) rec[-13] rec[-7] rec[-1]
        DETECTOR(1.5, 2.5, 0) rec[-16] rec[-9]
        DETECTOR(2.5, 0.5, 0) rec[-12] rec[-8]
        OBSERVABLE_INCLUDE(0) rec[-11] rec[-10] rec[-9] rec[-8] rec[-7] rec[-6]
        SHIFT_COORDS(0, 0, 1)
        TICK
        R 12 14 16 18 9 11 13 15
        TICK
        H 9 11 12 13 14 15 16 17 18
        SQRT_X 0
        TICK
        CZ 1 9 3 11 4 12 6 14 7 15 8 16
        TICK
        H 1 3 4 6 7 8
        TICK
        CZ 1 12 2 9 3 14 4 11 5 16 6 13 7 18 8 15
        TICK
        CZ 4 15 5 12 7 14
        TICK
        H 1 2 3 4 5 7 8 9
        TICK
        CZ 1 11 2 12 3 13 4 14 5 15 8 18
        TICK
        H 2 4 8 11 12 13 14 15 16 18
        TICK
        M 12 14 16 18 9 11 13 15
        DETECTOR(-0.5, 1.5, 0) rec[-13] rec[-4]
        DETECTOR(0.5, 0.5, 0) rec[-18] rec[-14] rec[-12] rec[-11] rec[-3]
        DETECTOR(0.5, 1.5, 0) rec[-19] rec[-8]
        DETECTOR(1.5, -0.5, 0) rec[-10] rec[-2]
        DETECTOR(1.5, 0.5, 0) rec[-18] rec[-7]
        DETECTOR(1.5, 1.5, 0) rec[-18] rec[-15] rec[-9] rec[-1]
        DETECTOR(1.5, 2.5, 0) rec[-17] rec[-6]
        DETECTOR(2.5, 1.5, 0) rec[-16] rec[-15] rec[-5]
        SHIFT_COORDS(0, 0, 1)
        TICK
        REPEAT 9 {
            R 12 14 16 18 9 11 13 15
            TICK
            H 4 6 8 9 11 12 13 14 15 16 18
            TICK
            CZ 1 9 3 11 4 12 6 14 7 15 8 16
            TICK
            H 1 3 4 5 6 7 8
            TICK
            CZ 1 12 2 9 3 14 4 11 5 16 6 13 7 18 8 15
            TICK
            CZ 4 15 5 12 7 14
            TICK
            H 1 2 3 4 5 7 8 9
            TICK
            CZ 1 11 2 12 3 13 4 14 5 15 8 18
            TICK
            H 2 4 8 11 12 13 14 15 16 18
            TICK
            M 12 14 16 18 9 11 13 15
            DETECTOR(-0.5, 1.5, 0) rec[-12] rec[-4]
            DETECTOR(0.5, 0.5, 0) rec[-11] rec[-3]
            DETECTOR(0.5, 1.5, 0) rec[-16] rec[-8]
            DETECTOR(1.5, -0.5, 0) rec[-10] rec[-2]
            DETECTOR(1.5, 0.5, 0) rec[-15] rec[-7]
            DETECTOR(1.5, 1.5, 0) rec[-9] rec[-1]
            DETECTOR(1.5, 2.5, 0) rec[-14] rec[-6]
            DETECTOR(2.5, 1.5, 0) rec[-13] rec[-5]
            SHIFT_COORDS(0, 0, 1)
            TICK
        }
        R 12 14 16 18 9 11 13 15
        TICK
        H 4 6 8 9 11 12 13 14 15 16 18
        TICK
        CZ 1 9 3 11 4 12 6 14 7 15 8 16
        TICK
        H 1 3 4 5 6 7 8
        TICK
        CZ 1 12 2 9 3 14 4 11 5 16 6 13 7 18 8 15
        TICK
        CZ 4 15 5 12 7 14
        TICK
        H 1 2 3 4 5 8 9
        TICK
        CZ 1 11 2 12 3 13 4 14 5 15 8 18
        TICK
        H 2 4 5 11 12 13 14 15 16 18
        TICK
        M 12 14 16 18 5 7 8 1 2 3 4 6 9 11 13 15
        DETECTOR(-0.5, 1.5, 0) rec[-20] rec[-4]
        DETECTOR(0.5, 0.5, 0) rec[-19] rec[-3]
        DETECTOR(0.5, 1.5, 0) rec[-24] rec[-16]
        DETECTOR(1.5, -0.5, 0) rec[-18] rec[-2]
        DETECTOR(1.5, 0.5, 0) rec[-23] rec[-15]
        DETECTOR(1.5, 1.5, 0) rec[-17] rec[-1]
        DETECTOR(1.5, 2.5, 0) rec[-22] rec[-14]
        DETECTOR(2.5, 1.5, 0) rec[-21] rec[-13]
        DETECTOR(-0.5, 1.5, 0) rec[-9] rec[-8] rec[-4]
        DETECTOR(0.5, 0.5, 0) rec[-9] rec[-7] rec[-6] rec[-3]
        DETECTOR(1.5, -0.5, 0) rec[-7] rec[-5] rec[-2]
        DETECTOR(1.5, 2.5, 0) rec[-14] rec[-12] rec[-10]
        DETECTOR(2.5, 1.5, 0) rec[-13] rec[-11] rec[-10]
        SHIFT_COORDS(0, 0, 1)
    """)

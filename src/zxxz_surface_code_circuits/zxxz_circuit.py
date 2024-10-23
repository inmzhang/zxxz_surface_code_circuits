from typing import Any, Dict, List, Optional, Union, cast

import stim
from midout.circuits.steps._patches import (
    DL,
    DR,
    UL,
    UR,
    rectangular_surface_code_patch,
)
from midout.gen._builder import AtLayer, Builder
from midout.gen._chunk import Chunk
from midout.gen._flow import PauliString, Flow
from midout.gen._flow_util import compile_chunks_into_circuit
from midout.gen._patch import Patch
from midout.gen._surface_code import checkerboard_basis
from midout.gen._tile import Tile


def build_zxxz_surface_code_round_circuit(
    patch: Patch,
    *,
    init_data_basis: Union[None, Dict[complex, str]] = None,
    measure_data_basis: Union[None, Dict[complex, str]] = None,
    save_layer: Any,
    out: Builder,
):
    init_data_basis = init_data_basis or {}
    measure_data_basis = measure_data_basis or {}

    out.gate("R", patch.measure_set)
    if init_data_basis:
        out.gate("R", patch.data_set)
    out.tick()
    if init_data_basis:
        out.gate("H", [q for q, basis in init_data_basis.items() if basis == "X"])
    out.tick()
    out.gate("H", patch.measure_set)
    out.tick()

    (num_layers,) = {len(tile.ordered_data_qubits) for tile in patch.tiles}
    for k in range(num_layers):
        out.gate2(
            "CZ",
            [
                (tile.measurement_qubit, cast(complex, tile.ordered_data_qubits[k]))
                for tile in patch.tiles
                if tile.ordered_data_qubits[k] is not None
            ],
        )
        out.tick()
        if k in [0, 2]:
            out.gate("H", patch.data_set)
            out.tick()

    out.gate("H", patch.measure_set)
    out.tick()
    if measure_data_basis:
        out.gate("H", [q for q, basis in measure_data_basis.items() if basis == "X"])
        out.tick()
    out.measure(patch.measure_set, basis="Z", save_layer=save_layer)
    if measure_data_basis:
        out.measure(patch.data_set, basis="Z", save_layer=save_layer)


def zxxz_surface_code_chunk(
    patch: Patch,
    *,
    init_data_basis: Union[None, Dict[complex, str]] = None,
    measure_data_basis: Union[None, Dict[complex, str]] = None,
    obs: Optional[PauliString] = None,
) -> Chunk:
    init_data_basis = init_data_basis or {}
    measure_data_basis = measure_data_basis or {}

    out = Builder.for_qubits(patch.used_set)
    save_layer = "solo"
    build_zxxz_surface_code_round_circuit(
        patch=patch,
        init_data_basis=init_data_basis,
        measure_data_basis=measure_data_basis,
        save_layer=save_layer,
        out=out,
    )

    def paulis(tile: Tile) -> PauliString:
        return PauliString(
            {k: v for k, v in zip(tile.ordered_data_qubits, "ZXXZ") if k is not None}
        )

    flows = []
    if not init_data_basis:
        flows.extend(
            Flow(
                center=tile.measurement_qubit,
                start=paulis(tile),
                measurement_indices=out.tracker.measurement_indices(
                    [AtLayer(tile.measurement_qubit, save_layer)]
                ),
            )
            for tile in patch.tiles
        )
    if not measure_data_basis:
        flows.extend(
            Flow(
                center=tile.measurement_qubit,
                end=paulis(tile),
                measurement_indices=out.tracker.measurement_indices(
                    [AtLayer(tile.measurement_qubit, save_layer)]
                ),
            )
            for tile in patch.tiles
        )
    flows.extend(
        Flow(
            center=tile.measurement_qubit,
            measurement_indices=out.tracker.measurement_indices(
                [AtLayer(tile.measurement_qubit, save_layer)]
            ),
        )
        for tile in patch.tiles
        if all(
            q is None or init_data_basis.get(q) == b
            for q, b in zip(tile.ordered_data_qubits, "ZXXZ")
        )
    )
    flows.extend(
        Flow(
            center=tile.measurement_qubit,
            measurement_indices=out.tracker.measurement_indices(
                [AtLayer(q, save_layer) for q in tile.used_set]
            ),
        )
        for tile in patch.tiles
        if all(
            q is None or measure_data_basis.get(q) == b
            for q, b in zip(tile.ordered_data_qubits, "ZXXZ")
        )
    )
    if obs is not None:
        start_obs = dict(obs.qubits)
        end_obs = dict(obs.qubits)
        for q in init_data_basis:
            if q in start_obs:
                if start_obs.pop(q) != init_data_basis[q]:
                    raise ValueError("wrong init basis for obs")
        measure_indices = []
        for q in measure_data_basis:
            if q in end_obs:
                if end_obs.pop(q) != measure_data_basis[q]:
                    raise ValueError("wrong measure basis for obs")
                measure_indices.extend(
                    out.tracker.measurement_indices([AtLayer(q, save_layer)])
                )

        flows.append(
            Flow(
                center=0,
                start=PauliString(start_obs),
                end=PauliString(end_obs),
                obs_index=0,
                measurement_indices=measure_indices,
            )
        )

    return Chunk(circuit=out.circuit, q2i=out.q2i, flows=flows)


def make_ztop_qubit_patch(*, distance: int) -> Patch:
    def order_func(m: complex) -> List[complex]:
        if checkerboard_basis(m) == "Z":
            return [UL, UR, DL, DR]
        else:
            return [UL, DL, UR, DR]

    return rectangular_surface_code_patch(
        width=distance,
        height=distance,
        top_basis="Z",
        right_basis="X",
        bot_basis="Z",
        left_basis="X",
        order_func=order_func,
    )


def make_zxxz_memory_experiment_chunks(
    *,
    distance: int,
    basis: str,
    rounds: int,
) -> List[Chunk]:
    qubit_patch = make_ztop_qubit_patch(distance=distance)
    zs = {q for q in qubit_patch.data_set if q.real == 0}
    xs = {q for q in qubit_patch.data_set if q.imag == 0}
    assert len(xs & zs) % 2 == 1
    # observable
    obs_z = PauliString({q: ("Z" if q.imag % 2 == 0 else "X") for q in zs})
    obs_x = PauliString({q: ("X" if q.real % 2 == 0 else "Z") for q in xs})
    obs = obs_z if basis == "Z" else obs_x
    # init/meas basis
    init_basis_z = {
        q: ("Z" if (q.real + q.imag) % 2 == 0 else "X") for q in qubit_patch.data_set
    }
    init_basis_x = {
        q: ("X" if (q.real + q.imag) % 2 == 0 else "Z") for q in qubit_patch.data_set
    }
    init_meas_basis = init_basis_z if basis == "Z" else init_basis_x
    assert rounds > 0
    if rounds == 1:
        return [
            zxxz_surface_code_chunk(
                qubit_patch,
                init_data_basis=init_meas_basis,
                measure_data_basis=init_meas_basis,
                obs=obs,
            )
        ]

    return [
        zxxz_surface_code_chunk(
            qubit_patch,
            init_data_basis=init_meas_basis,
            obs=obs,
        ),
        zxxz_surface_code_chunk(
            qubit_patch,
            obs=obs,
        ).with_repetitions(rounds - 2),
        zxxz_surface_code_chunk(
            qubit_patch,
            measure_data_basis=init_meas_basis,
            obs=obs,
        ),
    ]


def make_zxxz_memory_circuit(
    *,
    distance: int,
    basis: str,
    rounds: int,
) -> stim.Circuit:
    chunks = make_zxxz_memory_experiment_chunks(
        distance=distance, basis=basis, rounds=rounds
    )
    return compile_chunks_into_circuit(chunks)

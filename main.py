from midout import gen
from zxxz_surface_code_circuits import make_zxxz_memory_circuit


def main():
    d = 5
    circuit = make_zxxz_memory_circuit(
        distance=d,
        basis="Z",
        rounds=d,
    )
    print(circuit)
    noise_model = gen.NoiseModel.si1000(p=0.001)
    noisy_circuit = noise_model.noisy_circuit(circuit)
    dem = noisy_circuit.detector_error_model()
    assert len(dem.shortest_graphlike_error()) == d


if __name__ == "__main__":
    main()

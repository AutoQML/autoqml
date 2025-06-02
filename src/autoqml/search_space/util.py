from typing import Union, Callable
from collections.abc import Sequence
from optuna import Trial

import numpy as np

from autoqml.search_space import Configuration

from squlearn.encoding_circuit import (
    ChebyshevPQC,
    ChebyshevRx,
    ChebyshevTower,
    HighDimEncodingCircuit,
    HubregtsenEncodingCircuit,
    MultiControlEncodingCircuit,
    ParamZFeatureMap,
    YZ_CX_EncodingCircuit,
)
from squlearn.encoding_circuit.encoding_circuit_base import EncodingCircuitBase
from squlearn.observables import *
from squlearn.observables.observable_base import ObservableBase
from squlearn.kernel.lowlevel_kernel.kernel_matrix_base import KernelMatrixBase
from squlearn.kernel.lowlevel_kernel.projected_quantum_kernel import OuterKernelBase
from squlearn.kernel.lowlevel_kernel import ProjectedQuantumKernel, FidelityKernel
from squlearn.observables.observable_base import ObservableBase
from squlearn.util.executor import Executor


def get_encoding_circuit(
    encoding_circuit: str,
    num_qubits: int,
    num_features: int,
    num_repetitions: int,
    alpha: float,
    num_chebyshev: int,
) -> EncodingCircuitBase:
    num_layers = (
        num_features + num_qubits - 1
    ) // num_qubits * num_repetitions
    if encoding_circuit == "chebyshev_pqc":
        return ChebyshevPQC(
            num_qubits=num_qubits,
            num_features=num_features,
            num_layers=num_layers,
            alpha=alpha,
            nonlinearity='arctan',
        )
    elif encoding_circuit == "chebyshev_rx":
        return ChebyshevRx(
            num_qubits=num_qubits,
            num_features=num_features,
            num_layers=num_layers,
            alpha=alpha,
            nonlinearity='arctan',
        )
    elif encoding_circuit == "chebyshev_tower":
        return ChebyshevTower(
            num_qubits=num_qubits,
            num_features=num_features,
            num_chebyshev=num_chebyshev,
            alpha=alpha,
            num_layers=num_layers,
            nonlinearity='arctan',
        )
    elif encoding_circuit == "high_dim_encoding_circuit":
        return HighDimEncodingCircuit(
            num_qubits=num_qubits,
            num_features=num_features,
            num_layers=num_layers,
        )
    elif encoding_circuit == "hubregtsen_encoding_circuit":
        return HubregtsenEncodingCircuit(
            num_qubits=num_qubits,
            num_features=num_features,
            num_layers=num_layers,
        )
    elif encoding_circuit == "multi_control_encoding_circuit":
        return MultiControlEncodingCircuit(
            num_qubits=num_qubits,
            num_features=num_features,
            num_layers=num_layers,
        )
    elif encoding_circuit == "param_z_feature_map":
        return ParamZFeatureMap(
            num_qubits=num_qubits,
            num_features=num_features,
            num_layers=num_layers,
        )
    elif encoding_circuit == "yz_cx_encoding_circuit":
        return YZ_CX_EncodingCircuit(
            num_qubits=num_qubits,
            num_features=num_features,
            num_layers=num_layers,
        )
    else:
        raise ValueError(f"Encoding circuit {encoding_circuit} not supported")


def get_observable(
    observable: str,
    num_qubits: int,
    qubit: int,
) -> ObservableBase:
    if observable == "single_pauli_x":
        return SinglePauli(
            num_qubits=num_qubits,
            qubit=qubit,
            op_str="X",
        )
    elif observable == "single_pauli_y":
        return SinglePauli(
            num_qubits=num_qubits,
            qubit=qubit,
            op_str="Y",
        )
    elif observable == "single_pauli_z":
        return SinglePauli(
            num_qubits=num_qubits,
            qubit=qubit,
            op_str="Z",
        )
    elif observable == "summed_paulis_x":
        return SummedPaulis(
            num_qubits=num_qubits,
            op_str="X",
        )
    elif observable == "summed_paulis_y":
        return SummedPaulis(
            num_qubits=num_qubits,
            op_str="Y",
        )
    elif observable == "summed_paulis_z":
        return SummedPaulis(
            num_qubits=num_qubits,
            op_str="Z",
        )
    elif observable == "summed_paulis_xy":
        return SummedPaulis(
            num_qubits=num_qubits,
            op_str=("X", "Y"),
        )
    elif observable == "summed_paulis_yz":
        return SummedPaulis(
            num_qubits=num_qubits,
            op_str=("Y", "Z"),
        )
    elif observable == "summed_paulis_zx":
        return SummedPaulis(
            num_qubits=num_qubits,
            op_str=("Z", "X"),
        )
    elif observable == "summed_paulis_xyz":
        return SummedPaulis(
            num_qubits=num_qubits,
            op_str=("X", "Y", "Z"),
        )
    elif observable == "single_probability":
        return SingleProbability(num_qubits=num_qubits, qubit=qubit)
    elif observable == "summed_probabilities":
        return SummedProbabilities(num_qubits=num_qubits)
    elif observable == "ising_hamiltonian":
        return IsingHamiltonian(num_qubits=num_qubits, I="N")
    elif observable == "ising_hamiltonian_transverse":
        return IsingHamiltonian(
            num_qubits=num_qubits, I="N", Z="N", X="F", ZZ="F"
        )
    else:
        raise ValueError(f"Observable {observable} not supported")


def get_quantum_kernel(
    quantum_kernel: str,
    encoding_circuit: EncodingCircuitBase,
    executor: Executor,
    measurement: Union[str, ObservableBase, list],
    outer_kernel: Union[str, OuterKernelBase],
    parameter_seed: Union[int, None] = 0,
) -> KernelMatrixBase:
    if quantum_kernel == "projected_quantum_kernel":
        return ProjectedQuantumKernel(
            encoding_circuit=encoding_circuit,
            executor=executor,
            measurement=measurement,
            outer_kernel=outer_kernel,
            parameter_seed=parameter_seed,
        )
    elif quantum_kernel == "fidelity_quantum_kernel":
        return FidelityKernel(
            encoding_circuit=encoding_circuit,
            executor=executor,
            parameter_seed=parameter_seed,
        )
    else:
        raise ValueError(f"Quantum Kernel {quantum_kernel} not supported")


def sample_quantum_kernel_configuration(
    trial: Trial, defaults: Configuration, fullname: Callable
) -> Configuration:
    """
    Generates a configuration for a quantum kernel.

    Args:
        trial (Trial): The trial object used for suggesting values.
        defaults (Configuration): The default configuration values.
        fullname (Callable): The function used to generate the full name of a
            configuration parameter.

    Returns:
        Configuration: The configuration for the quantum kernel.

    Raises:
        ValueError: If the quantum kernel is not supported.
    """
    def _get_default_values(
        trial: Trial, fullname: str, defaults: Configuration
    ):
        """ Helper function to handle default sequences"""
        return (
            trial.suggest_categorical(fullname, defaults[fullname])
            if isinstance(defaults[fullname], Sequence) and
            not isinstance(defaults[fullname],
                           (str, bytes)) else defaults[fullname]
        )

    quantum_kernel = (
        _get_default_values(trial, fullname("quantum_kernel"), defaults) if
        fullname("quantum_kernel") in defaults else trial.suggest_categorical(
            fullname("quantum_kernel"),
            ["projected_quantum_kernel", "fidelity_quantum_kernel"],
        )
    )

    parameter_seed = (
        _get_default_values(trial, fullname("parameter_seed"), defaults)
        if fullname("parameter_seed") in defaults else
        trial.suggest_int(fullname("parameter_seed"), 0, 2**32 - 1)
    )

    if quantum_kernel == "projected_quantum_kernel":
        return {
            "quantum_kernel": "projected_quantum_kernel",
            "measurement":
                (
                    _get_default_values(
                        trial, fullname("measurement"), defaults
                    ) if fullname("measurement") in defaults else
                    trial.suggest_categorical(
                        fullname("measurement"),
                        ["X", "Y", "Z", "XY", "XZ", "YZ", "XYZ"],
                    )
                ),
            "outer_kernel":
                (
                    _get_default_values(
                        trial, fullname("outer_kernel"), defaults
                    ) if fullname("outer_kernel") in defaults else
                    trial.suggest_categorical(
                        fullname("outer_kernel"),
                        [
                            "Gaussian",
                            "Matern",
                            "ExpSineSquared",
                            "RationalQuadratic",
                            "DotProduct",
                            "PairwiseKernel",
                        ],
                    )
                ),
            "parameter_seed": parameter_seed,
        }
    elif quantum_kernel == "fidelity_quantum_kernel":
        return {
            "quantum_kernel": "fidelity_quantum_kernel",
            "parameter_seed": parameter_seed,
        }
    else:
        raise ValueError(f"Quantum Kernel {quantum_kernel} not supported")


def sample_encoding_circuit_configuration(
    trial: Trial, defaults: Configuration, fullname: Callable
) -> Configuration:
    """
    Generates a configuration for an encoding circuit.

    Args:
        trial (Trial): The trial object used for suggesting values.
        defaults (Configuration): The default configuration values.
        fullname (Callable): The function used to generate the full name of a
            configuration parameter.

    Returns:
        Configuration: The configuration for the encoding circuit.

    Raises:
        ValueError: If the encoding circuit is not supported.

    """
    def _get_default_values(
        trial: Trial, fullname: str, defaults: Configuration
    ):
        """ Helper function to handle default sequences"""
        return (
            trial.suggest_categorical(fullname, defaults[fullname])
            if isinstance(defaults[fullname], Sequence) and
            not isinstance(defaults[fullname],
                           (str, bytes)) else defaults[fullname]
        )

    config = {}
    config["encoding_circuit"] = (
        _get_default_values(trial, fullname("encoding_circuit"), defaults)
        if fullname("encoding_circuit") in defaults else
        trial.suggest_categorical(
            fullname("encoding_circuit"),
            [
                "chebyshev_pqc",
                "chebyshev_rx",
                "chebyshev_tower",
                "high_dim_encoding_circuit",
                "hubregtsen_encoding_circuit",
                "multi_control_encoding_circuit",
                "param_z_feature_map",
                "yz_cx_encoding_circuit",
            ],
        )
    )

    config["num_repetitions"] = (
        _get_default_values(trial, fullname("num_repetitions"), defaults)
        if fullname("num_repetitions") in defaults else
        trial.suggest_categorical(fullname("num_repetitions"), [1, 2, 3])
    )

    if config["encoding_circuit"] in [
        "chebyshev_pqc",
        "chebyshev_rx",
        "chebyshev_tower",
    ]:
        config["chebyshev_alpha"] = (
            _get_default_values(trial, fullname("chebyshev_alpha"), defaults)
            if fullname("chebyshev_alpha") in defaults else
            trial.suggest_float(fullname("chebyshev_alpha"), 0.01, 10.0)
        )
        if config["encoding_circuit"] == "chebyshev_tower":
            config["num_chebyshev"] = (
                _get_default_values(
                    trial, fullname("num_chebyshev"), defaults
                ) if fullname("num_chebyshev") in defaults else trial.
                suggest_categorical(fullname("num_chebyshev"), [1, 2, 3])
            )

    return config


def sample_observable_configuration(
    trial: Trial, defaults: Configuration, fullname: Callable, num_qubits: int
) -> Configuration:
    """
    Generates a configuration for an observable.

    Args:
        trial (Trial): The trial object used for suggesting values.
        defaults (Configuration): The default configuration values.
        fullname (Callable): The function used to generate the full name of a
            configuration parameter.
        num_qubits (int): The number of qubits in the system.

    Returns:
        Configuration: The configuration for the observable.

    Raises:
        ValueError: If the observable is not supported.

    """
    def _get_default_values(
        trial: Trial, fullname: str, defaults: Configuration
    ):
        """ Helper function to handle default sequences"""
        return (
            trial.suggest_categorical(fullname, defaults[fullname])
            if isinstance(defaults[fullname], Sequence) and
            not isinstance(defaults[fullname],
                           (str, bytes)) else defaults[fullname]
        )

    observable = (
        defaults[fullname("observable")]
        if fullname("observable") in defaults else trial.suggest_categorical(
            fullname("observable"),
            [
                "single_pauli_x",
                "single_pauli_y",
                "single_pauli_z",
                "summed_paulis_x",
                "summed_paulis_y",
                "summed_paulis_z",
                "summed_paulis_xy",
                "summed_paulis_yz",
                "summed_paulis_zx",
                "summed_paulis_xyz",
                "single_probability",
                "summed_probabilities",
                "ising_hamiltonian",
                "ising_hamiltonian_transverse",
            ],
        )
    )

    if observable.startswith("single"):
        qubit = (
            _get_default_values(trial, fullname("observable_qubit"), defaults)
            if fullname("observable_qubit") in defaults else
            trial.suggest_int(fullname("observable_qubit"), 0, num_qubits - 1)
        )
        return {
            "observable": observable,
            "observable_qubit": qubit,
        }

    return {"observable": observable}

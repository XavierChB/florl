from abc import ABC, abstractmethod
import struct
from typing import List

from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.typing import (
    Code,
    Status,
    GetParametersIns,
    GetParametersRes,
    Parameters,
    NDArrays,
)


class Knowledge(ABC):
    """Represents abstract information learnt by a RL algorithm

    Examples include networks for the world model, policy or value functions.
    """

    # Serialisation hack
    # Ideally this should correspond to a gRPC class instead
    SEP_CHAR = "|"

    def __init__(self, modules: List[str]) -> None:
        super().__init__()
        self._modules = modules
        self._modules_registry = {id_: i for i, id_ in enumerate(self._modules)}

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        modules = ins.config.get("modules", self._modules)
        all_parameters = [self.get_module_parameters(x, ins) for x in modules]

        # Status Validation
        for parameter_res in all_parameters:
            if parameter_res.status != Code.OK:
                return parameter_res

        message = ",".join([x for x in modules])
        return GetParametersRes(
            status=Status(
                code=Code.OK,
                message=f"Sending {message}",
            ),
            parameters=Parameters(
                tensor_type=Knowledge.SEP_CHAR.join(
                    [x.parameters.tensor_type for x in all_parameters]
                ),
                tensors=[],
            ),
        )

    def get_module_parameters(
        self, id_: str, ins: GetParametersIns
    ) -> GetParametersRes:
        # id_ exists
        if not id_ in self._modules_registry:
            return GetParametersRes(
                Status(
                    code=Code.GET_PARAMETERS_NOT_IMPLEMENTED,
                    message=f"Module {id_} does not exist in registry.",
                )
            )

        parameter_res = self._get_module_parameters(id_, ins)
        # Throw errors
        if parameter_res.status != Code.OK:
            return parameter_res
        # Prepend id_ to tensors
        id_: int = self._modules_registry[id_]
        parameter_res.parameters.tensors = [
            struct.pack(">I", id_) + tensor
            for tensor in parameter_res.parameters.tensors
        ]
        return parameter_res

    def set_parameters(self, ins: Parameters) -> None:
        # Deserialisation
        i = 0
        tensor_types = ins.tensor_type.split(Knowledge.SEP_CHAR)
        tensor_buffer = []
        previous_id = None
        for id_tensor in ins.tensors:
            id_ = struct.unpack(">I", id_tensor[:4])
            tensor = id_tensor[4:]
            if previous_id is None or previous_id == id_:
                tensor_buffer.append(tensor)
            else:
                parameters = Parameters(
                    tensors=tensor_buffer, tensor_types=tensor_types[i]
                )
                self._set_module_parameters(self._modules[id_], parameters)
                i, tensor_buffer = i + 1, []
            previous_id = id_

    @abstractmethod
    def _get_module_parameters(
        self, id_: str, ins: GetParametersIns
    ) -> GetParametersRes:
        raise NotImplementedError

    @abstractmethod
    def _set_module_parameters(self, id_: str, ins: Parameters) -> None:
        raise NotImplementedError


class NumPyKnowledge(Knowledge, ABC):
    """Knowledge, where individual module parameters can be transformed into numpy parameters"""

    def _get_module_parameters(
        self, id_: str, ins: GetParametersIns
    ) -> GetParametersRes:
        numpy_parameters = self._get_module_parameters_numpy(id_, ins)
        return GetParametersRes(
            status=Code.OK,
            parameters=ndarrays_to_parameters(numpy_parameters)
        )

    def _set_module_parameters(self, id_: str, ins: Parameters) -> None:
        return self._set_module_parameters_numpy(id_, parameters_to_ndarrays(ins))

    @abstractmethod
    def _get_module_parameters_numpy(self, id_: str, ins: GetParametersIns) -> NDArrays:
        raise NotImplementedError

    @abstractmethod
    def _set_module_parameters_numpy(self, id_: str, ins: Parameters) -> None:
        raise NotImplementedError

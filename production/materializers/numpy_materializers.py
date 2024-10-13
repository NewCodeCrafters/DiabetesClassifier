import numpy as np
from zenml.io import fileio
from zenml.enums import ArtifactType
from zenml.materializers.base_materializer import BaseMaterializer
from typing import Type

class NumpyMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = (np.ndarray,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA  # Corrected to a valid artifact type

    def load(self, data_type: Type[np.ndarray]) -> np.ndarray:
        """Reads a NumPy array from a .npy file."""
        filepath = self.uri + ".npy"
        with fileio.open(filepath, "rb") as f:
            return np.load(f)
        # with self.artifact_store.open(os.path.join(self.uri, 'data.txt'), 'r') as f:
        #     name = f.read()
        # return MyObj(name=name)

    def save(self, array: np.ndarray) -> None:
        """Writes a NumPy array to a .npy file."""
        filepath = self.uri + ".npy"
        with fileio.open(filepath, "wb") as f:
            np.save(f, array)

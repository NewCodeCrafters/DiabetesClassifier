import joblib
from zenml.io import fileio
from zenml.materializers.base_materializer import BaseMaterializer
from sklearn.base import ClassifierMixin
from zenml.enums import ArtifactType
from typing import Type

class ModelMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = (ClassifierMixin,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.MODEL  # Corrected to a valid artifact type

    def load(self, data_type: Type[ClassifierMixin]) -> ClassifierMixin:
        """Loads a RandomForestClassifier model from a .joblib file."""
        filepath = self.uri + ".joblib"
        with fileio.open(filepath, "rb") as f:
            return joblib.load(f)

    def save(self, model: ClassifierMixin) -> None:
        """Saves a RandomForestClassifier model to a .joblib file."""
        filepath = self.uri + ".joblib"
        with fileio.open(filepath, "wb") as f:
            joblib.dump(model, f)

from materializers.model_materializers import ModelMaterializer
from materializers.numpy_materializers import NumpyMaterializer
import numpy as np
from sklearn.base import ClassifierMixin
from pipeline import diabetes_classifier_pipeline
from zenml.materializers.materializer_registry import materializer_registry

materializer_registry.register_and_overwrite_type(key=np.ndarray, type_=NumpyMaterializer)
materializer_registry.register_and_overwrite_type(key=ClassifierMixin, type_=ModelMaterializer)


if __name__ == "__main__":
    run = diabetes_classifier_pipeline()

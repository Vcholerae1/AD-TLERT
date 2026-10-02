"""Top-level package for differentiable time-lapse ERT on Torch."""

from importlib.metadata import PackageNotFoundError, version

from adtlert.fem import P1ElementData, build_p1_element_data, p1_shape_functions
from adtlert.forward import (
    ERT2p5DApparentResistivityFunction,
    ERTForward2p5D,
    ERTForwardModeling,
    ForwardResponse,
    apparent_resistivity_autograd,
)
from adtlert.inversion import (
    ERTInversion,
    InversionConfig,
    TimeLapseERTInversion,
    WindowedTimeLapseERTInversion,
)
from adtlert.mesh import Mesh
from adtlert.survey import Survey

try:
    __version__ = version("adtlert")
except PackageNotFoundError:  # pragma: no cover - source tree without installation
    __version__ = "0+unknown"

__all__ = [
    "ERT2p5DApparentResistivityFunction",
    "ERTForward2p5D",
    "ERTForwardModeling",
    "ERTInversion",
    "ForwardResponse",
    "InversionConfig",
    "Mesh",
    "P1ElementData",
    "Survey",
    "TimeLapseERTInversion",
    "WindowedTimeLapseERTInversion",
    "__version__",
    "apparent_resistivity_autograd",
    "build_p1_element_data",
    "p1_shape_functions",
]

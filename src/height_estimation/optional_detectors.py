"""Small helpers for loading optional model-backed detectors on demand."""

from __future__ import annotations

import importlib
from typing import Any, Callable


class OptionalDetectorUnavailable(RuntimeError):
    """Raised when an optional detector cannot be constructed."""


def import_optional_module(module_name: str) -> Any:
    """Import an optional module while preserving a useful failure message."""
    try:
        return importlib.import_module(module_name)
    except ImportError as error:
        raise OptionalDetectorUnavailable(
            f"optional module {module_name!r} is unavailable: {error}"
        ) from error


class LazyDetector:
    """Construct one detector instance on first use and reuse it thereafter."""

    def __init__(
        self,
        factory: Callable[[], object],
        *,
        name: str = "optional detector",
    ) -> None:
        if not callable(factory):
            raise TypeError("detector factory must be callable")
        if not name.strip():
            raise ValueError("detector name must not be empty")
        self._factory = factory
        self._instance: object | None = None
        self._initialisation_error: OptionalDetectorUnavailable | None = None
        self.name = name

    @property
    def loaded(self) -> bool:
        return self._instance is not None

    def _load(self) -> object:
        if self._initialisation_error is not None:
            raise self._initialisation_error
        if self._instance is None:
            try:
                instance = self._factory()
            except Exception as error:
                self._initialisation_error = OptionalDetectorUnavailable(
                    f"{self.name} could not be initialised: {error}"
                )
                raise self._initialisation_error from error
            if instance is None:
                self._initialisation_error = OptionalDetectorUnavailable(
                    f"{self.name} factory returned no detector"
                )
                raise self._initialisation_error
            self._instance = instance
        return self._instance

    def detect(self, image: object) -> object:
        return self.invoke("detect", image)

    def invoke(self, method_name: str, *args: object, **kwargs: object) -> object:
        detector = self._load()
        method = getattr(detector, method_name, None)
        if method is None and method_name == "detect":
            method = detector
        if not callable(method):
            raise TypeError(
                f"{self.name} must expose a callable {method_name} method"
            )
        return method(*args, **kwargs)

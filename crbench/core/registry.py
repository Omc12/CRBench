"""
Task and Adapter Registry for CRBench.
Allows extensible registration of benchmark tasks and context representations.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, Optional, Type


class Registry:
    """Central registry for Tasks and Adapters."""
    _adapters: Dict[str, Type[Any]] = {}
    _tasks: Dict[str, Type[Any]] = {}

    @classmethod
    def register_adapter(cls, name: str) -> Callable[[Type[Any]], Type[Any]]:
        def decorator(subclass: Type[Any]) -> Type[Any]:
            cls._adapters[name.lower()] = subclass
            return subclass
        return decorator

    @classmethod
    def register_task(cls, name: str) -> Callable[[Type[Any]], Type[Any]]:
        def decorator(subclass: Type[Any]) -> Type[Any]:
            cls._tasks[name.lower()] = subclass
            return subclass
        return decorator

    @classmethod
    def get_adapter(cls, name: str) -> Type[Any]:
        normalized = name.lower()
        if normalized not in cls._adapters:
            available = list(cls._adapters.keys())
            raise KeyError(f"Adapter '{name}' not found in registry. Available adapters: {available}")
        return cls._adapters[normalized]

    @classmethod
    def get_task(cls, name: str) -> Type[Any]:
        normalized = name.lower()
        if normalized not in cls._tasks:
            available = list(cls._tasks.keys())
            raise KeyError(f"Task '{name}' not found in registry. Available tasks: {available}")
        return cls._tasks[normalized]

    @classmethod
    def list_adapters(cls) -> Dict[str, Type[Any]]:
        return dict(cls._adapters)

    @classmethod
    def list_tasks(cls) -> Dict[str, Type[Any]]:
        return dict(cls._tasks)

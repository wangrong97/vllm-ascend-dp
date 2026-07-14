# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Breakable ACL graph capture/replay.

This is an NPU port of :mod:`vllm.compilation.breakable_cudagraph`. It drives
``torch.npu.NPUGraph`` instead of ``torch.cuda.CUDAGraph`` and is meant to be
used by :class:`vllm_ascend.compilation.acl_graph.ACLGraphWrapper`.

The idea (inspired by sgl-project/sglang#19102): instead of capturing the whole
forward as one monolithic ACL graph, a single capture context drives the whole
forward and intercepts attention custom ops at the dispatcher to end the
current stream-capture segment, run the op eagerly on the capture stream, and
resume capture. The captured artifact is a list of zero-arg callables -- the
bound ``NPUGraph.replay`` for graph segments, or the user fn for eager segments
-- replayed in order at inference time.

This is how vllm-ascend keeps the attention part "out of the ACL graph":
GEMMs / RMSNorms / MoE stay captured, while attention runs eagerly between
captured segments. Enabled via ``VLLM_ASCEND_ATTN_EAGER_BREAK`` (see
:mod:`vllm_ascend.envs`).
"""

from __future__ import annotations

import functools
import threading
from collections.abc import Callable
from typing import Any, TypeVar

import torch
from vllm.logger import init_logger

from ..utils import weak_ref_tensors

_logger = init_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class BreakableACLGraphCapture:
    """Stream-capture context that supports eager breaks via :meth:`add_eager`.

    Usage::

        cap = BreakableACLGraphCapture(pool=...)
        with cap:
            output = model(*static_inputs)
        # Later, after copying new inputs into the static buffers:
        cap.replay()
        # Output tensors live at the same addresses as during capture.

    Thread-local: only one capture may be active per thread.
    """

    _tls = threading.local()

    @classmethod
    def current(cls) -> BreakableACLGraphCapture | None:
        return getattr(cls._tls, "active", None)

    @classmethod
    def is_active(cls) -> bool:
        return cls.current() is not None

    def __init__(self, pool: Any | None = None) -> None:
        self.pool = pool
        self.segments: list[Callable[[], Any]] = []
        self._num_graphs: int = 0
        self._num_eager_breaks: int = 0
        self._current_graph: torch.npu.NPUGraph | None = None
        self._capturing: bool = False

    # --- context manager protocol ----------------------------------------

    def __enter__(self) -> BreakableACLGraphCapture:
        if getattr(BreakableACLGraphCapture._tls, "active", None) is not None:
            raise RuntimeError("Nested BreakableACLGraphCapture is not supported.")
        BreakableACLGraphCapture._tls.active = self
        self._begin_segment()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self._end_segment()
        finally:
            BreakableACLGraphCapture._tls.active = None

    # --- segment management ----------------------------------------------

    def _begin_segment(self) -> None:
        assert not self._capturing
        g = torch.npu.NPUGraph()
        # Same capture API as torch.cuda.CUDAGraph (verified on torch_npu).
        g.capture_begin(pool=self.pool)
        self._current_graph = g
        self._capturing = True

    def _end_segment(self) -> None:
        if not self._capturing:
            return
        assert self._current_graph is not None
        self._current_graph.capture_end()
        self.segments.append(self._current_graph.replay)
        self._num_graphs += 1
        self._current_graph = None
        self._capturing = False

    def add_eager(self, fn: Callable[[], Any]) -> Any:
        """End the current capture segment, run ``fn`` eagerly on the capture
        stream, record ``fn`` for replay, and start a new segment.

        Returns whatever ``fn`` returned during this (capture-time) call.
        Replay does not return values; callers must propagate any downstream
        dependencies via static output buffers.
        """
        self._end_segment()
        result = fn()
        self.segments.append(fn)
        self._num_eager_breaks += 1
        self._begin_segment()
        return result

    # --- replay ----------------------------------------------------------

    def replay(self) -> None:
        for seg in self.segments:
            seg()

    # --- introspection ---------------------------------------------------

    @property
    def num_graphs(self) -> int:
        return self._num_graphs

    @property
    def num_eager_breaks(self) -> int:
        return self._num_eager_breaks

    def __repr__(self) -> str:
        return f"BreakableACLGraphCapture(graphs={self.num_graphs}, eager_breaks={self.num_eager_breaks})"


def eager_break_during_capture(fn: F) -> F:
    """Decorator that turns an attention kernel into a "break point" for the
    breakable ACL graph capture.

    When the decorated function is invoked outside of a
    :class:`BreakableACLGraphCapture` context (eager runs, warmup, profiling,
    prefill, or when the feature is disabled), it executes normally.

    When invoked inside an active capture, it ends the current ACL graph
    segment, runs the function eagerly on the capture stream, records the
    callable for replay, and starts a fresh segment -- so the decorated op is
    **never captured into the ACL graph** (it stays "out of graph").

    .. note::
        Unlike vLLM's CUDA original, we DO break under ``CUDAGraphMode.FULL``
        too -- that is exactly the intent here: keep attention out of the
        FULL decode graph while the rest of the model stays captured.

    **In-place output buffer required.** Decorated ops must write into a
    caller-provided output tensor; a fresh tensor returned by ``fn`` would
    change address each replay and break downstream graph segments.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        capture = BreakableACLGraphCapture.current()
        if capture is None or not capture._capturing:
            return fn(*args, **kwargs)

        # Weak-ref tensor args: strong refs inside the replay lambda would pin
        # ACL-graph-pool slots across batch descriptors. The graph owns the
        # slot, so the weak ref is safe to deref on replay.
        weak_args = tuple(weak_ref_tensors(a) if isinstance(a, torch.Tensor) else a for a in args)
        weak_kwargs = {k: weak_ref_tensors(v) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
        _logger.debug("Breakable ACL graph: eager-breaking out of capture for %s", fn)
        return capture.add_eager(lambda: fn(*weak_args, **weak_kwargs))

    return wrapper  # type: ignore[return-value]

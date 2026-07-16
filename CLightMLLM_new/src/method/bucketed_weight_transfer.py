# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Bucketed tensor transfer for student vLLM weight sync.

This is the small subset of VERL's bucketed weight transfer that the OPD
student server uses. Keeping it local makes the Lightning OPD path independent
from an external ``verl_new`` checkout.
"""

from __future__ import annotations

import gc
import inspect
import os
from collections.abc import AsyncIterator, Callable, Iterable
from multiprocessing import shared_memory
from typing import Any, TypedDict

import torch
import zmq
from torch.multiprocessing.reductions import reduce_tensor


class TensorMetadata(TypedDict):
    name: str
    shape: torch.Size
    dtype: torch.dtype
    offset: int
    handle: tuple[Any, Any] | None


async def _ensure_async_iterator(items: Any) -> AsyncIterator[Any]:
    if hasattr(items, "__aiter__"):
        async for item in items:
            yield item
        return
    for item in items:
        if inspect.isawaitable(item):
            item = await item
        yield item


def _current_cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("Bucketed CUDA IPC weight sync requires CUDA.")
    return torch.device("cuda", torch.cuda.current_device())


def _device_synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _device_cleanup() -> None:
    if torch.cuda.is_available():
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()


def rebuild_ipc(handle: tuple[Callable[..., Any], tuple[Any, ...]], device_id: int | None = None) -> torch.Tensor:
    func, args = handle
    list_args = list(args)
    if device_id is not None and len(list_args) > 6:
        list_args[6] = device_id
    return func(*list_args)


def create_shared_memory(size: int, name: str) -> shared_memory.SharedMemory:
    try:
        return shared_memory.SharedMemory(name=name, create=True, size=size)
    except FileExistsError:
        shm = shared_memory.SharedMemory(name=name)
        if shm.size < size:
            raise RuntimeError(f"Stale shm segment {name!r}: expected {size} bytes, got {shm.size}")
        return shm


def rebuild_shared_memory(name: str, size: int, dtype: torch.dtype = torch.uint8) -> tuple[torch.Tensor, shared_memory.SharedMemory]:
    shm = shared_memory.SharedMemory(name=name)
    return torch.frombuffer(shm.buf[:size], dtype=dtype), shm


class BucketedWeightSender:
    def __init__(self, zmq_handle: str, bucket_size_mb: int = 512, use_shm: bool = False) -> None:
        self.zmq_handle = zmq_handle
        self.bucket_size_mb = int(bucket_size_mb)
        self.bucket_size = self.bucket_size_mb << 20
        self.use_shm = bool(use_shm)
        self.zmq_context = zmq.Context.instance()
        self.socket: zmq.Socket | None = None
        self.buffer: torch.Tensor | None = None
        self.shm: shared_memory.SharedMemory | None = None

    async def async_send_weights(self, weights: Iterable[tuple[str, torch.Tensor]] | AsyncIterator[tuple[str, torch.Tensor]]) -> None:
        try:
            self._init_socket()
            self._init_buffer()

            offset = 0
            bucket_meta: dict[str, TensorMetadata] = {}
            async for name, weight in _ensure_async_iterator(weights):
                if self.buffer is None or self.socket is None:
                    raise RuntimeError("BucketedWeightSender is not initialized.")

                if offset + weight.nbytes > self.bucket_size and bucket_meta:
                    _device_synchronize()
                    self.socket.send_pyobj({"bucket_meta": bucket_meta, "is_last": False})
                    self.socket.recv()
                    bucket_meta = {}
                    offset = 0

                if offset + weight.nbytes > self.bucket_size:
                    if self.use_shm:
                        raise RuntimeError(
                            f"Weight {name}({tuple(weight.shape)}, {weight.dtype}) is too large for "
                            f"bucket_size_mb={self.bucket_size_mb}."
                        )
                    self._direct_send_large_weight(name, weight)
                    continue

                bucket_meta[name] = {
                    "name": name,
                    "shape": weight.shape,
                    "dtype": weight.dtype,
                    "offset": offset,
                    "handle": None,
                }
                self.buffer[offset : offset + weight.nbytes].copy_(weight.view(-1).view(torch.uint8), non_blocking=True)
                offset += weight.nbytes

            _device_synchronize()
            if self.socket is None:
                raise RuntimeError("BucketedWeightSender socket is not initialized.")
            self.socket.send_pyobj({"bucket_meta": bucket_meta, "is_last": True})
            self.socket.recv()
        finally:
            self._cleanup()

    def _init_socket(self) -> None:
        if self.zmq_handle.startswith("ipc://"):
            ipc_path = self.zmq_handle[len("ipc://") :]
            try:
                os.remove(ipc_path)
            except OSError:
                pass
        self.socket = self.zmq_context.socket(zmq.REQ)
        self.socket.bind(self.zmq_handle)

    def _init_buffer(self) -> None:
        if self.socket is None:
            raise RuntimeError("BucketedWeightSender socket is not initialized.")

        if self.use_shm:
            import uuid

            shm_name = f"clight_weights_{uuid.uuid4().hex}"
            self.shm = create_shared_memory(self.bucket_size, shm_name)
            self.buffer = torch.frombuffer(self.shm.buf, dtype=torch.uint8)
            self.socket.send_pyobj({"name": shm_name, "size": self.bucket_size})
        else:
            self.buffer = torch.empty(self.bucket_size, dtype=torch.uint8, device=_current_cuda_device())
            self.socket.send_pyobj(reduce_tensor(self.buffer))

        self.socket.recv()

    def _direct_send_large_weight(self, name: str, weight: torch.Tensor) -> None:
        if self.socket is None:
            raise RuntimeError("BucketedWeightSender socket is not initialized.")
        bucket_meta: dict[str, TensorMetadata] = {
            name: {
                "name": name,
                "shape": weight.shape,
                "dtype": weight.dtype,
                "offset": 0,
                "handle": reduce_tensor(weight),
            }
        }
        self.socket.send_pyobj({"bucket_meta": bucket_meta, "is_last": False})
        self.socket.recv()

    def _cleanup(self) -> None:
        if self.socket is not None:
            self.socket.close()
            self.socket = None
        if self.zmq_handle.startswith("ipc://"):
            ipc_path = self.zmq_handle[len("ipc://") :]
            try:
                os.remove(ipc_path)
            except OSError:
                pass
        self.buffer = None
        if self.shm is not None:
            self.shm.close()
            self.shm.unlink()
            self.shm = None
        gc.collect()
        _device_cleanup()


class BucketedWeightReceiver:
    def __init__(self, zmq_handle: str, device: torch.device, use_shm: bool = False) -> None:
        self.zmq_handle = zmq_handle
        self.device = device
        self.use_shm = bool(use_shm)
        self.zmq_context = zmq.Context.instance()
        self.socket: zmq.Socket | None = None
        self.buffer: torch.Tensor | None = None
        self.shm: shared_memory.SharedMemory | None = None

    def receive_weights(self, on_bucket_received: Callable[[list[tuple[str, torch.Tensor]]], None]) -> None:
        try:
            self._init_socket()
            self._init_buffer()

            while True:
                if self.socket is None or self.buffer is None:
                    raise RuntimeError("BucketedWeightReceiver is not initialized.")
                metadata = self.socket.recv_pyobj()
                weights: list[tuple[str, torch.Tensor]] = []
                tensor: torch.Tensor | None = None
                for name, meta in metadata["bucket_meta"].items():
                    shape, dtype, offset, handle = meta["shape"], meta["dtype"], meta["offset"], meta["handle"]
                    if handle is not None:
                        tensor = rebuild_ipc(handle, self.device.index)
                    else:
                        size = dtype.itemsize * shape.numel()
                        tensor = self.buffer[offset : offset + size].view(dtype=dtype).view(shape)
                        if self.use_shm:
                            tensor = tensor.to(self.device)
                    weights.append((name, tensor))

                on_bucket_received(weights)
                _device_synchronize()
                self.socket.send(b"")
                del weights, tensor
                if metadata["is_last"]:
                    break
        finally:
            self._cleanup()

    def _init_socket(self) -> None:
        self.socket = self.zmq_context.socket(zmq.REP)
        self.socket.connect(self.zmq_handle)

    def _init_buffer(self) -> None:
        if self.socket is None:
            raise RuntimeError("BucketedWeightReceiver socket is not initialized.")
        comm_metadata = self.socket.recv_pyobj()
        if self.use_shm:
            self.buffer, self.shm = rebuild_shared_memory(comm_metadata["name"], comm_metadata["size"], dtype=torch.uint8)
        else:
            self.buffer = rebuild_ipc(comm_metadata, self.device.index)
            if self.buffer.dtype != torch.uint8:
                raise RuntimeError(f"Expected uint8 IPC buffer, got {self.buffer.dtype}.")
        self.socket.send(b"")

    def _cleanup(self) -> None:
        if self.socket is not None:
            self.socket.close()
            self.socket = None
        _device_synchronize()
        self.buffer = None
        if self.shm is not None:
            self.shm.close()
            self.shm = None
        gc.collect()
        _device_cleanup()

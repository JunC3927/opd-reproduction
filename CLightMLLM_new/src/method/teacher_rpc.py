import pickle
import socket
import struct
from typing import Any


HEADER = struct.Struct("!Q")


def _recv_exact(sock: socket.socket, nbytes: int) -> bytes:
    chunks = []
    remaining = nbytes
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("Socket closed while receiving RPC payload.")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_message(sock: socket.socket, message: Any) -> None:
    payload = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(HEADER.pack(len(payload)))
    sock.sendall(payload)


def recv_message(sock: socket.socket) -> Any:
    header = _recv_exact(sock, HEADER.size)
    (payload_size,) = HEADER.unpack(header)
    payload = _recv_exact(sock, payload_size)
    return pickle.loads(payload)


def rpc_call(host: str, port: int, message: Any, timeout: float) -> Any:
    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.settimeout(timeout)
        send_message(sock, message)
        return recv_message(sock)

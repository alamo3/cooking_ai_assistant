"""Answer to a name the cook can remember.

Nobody wants to type 192.168.40.252 into a tablet, and the address changes the moment the
router hands out a new lease. This advertises a fixed mDNS name — kitchen.local by default —
pointing at whichever LAN address the machine currently has.

mDNS needs no router configuration and no DNS server, but resolution is up to the client:
Windows, macOS and iOS do it natively; Android has supported it since 12 and Chrome/WebView
generally follow, but older tablets may not. When it does not work, a DHCP reservation plus
the router's own DNS is the fallback, and the IP always works.
"""
from __future__ import annotations

import logging
import os
import socket
from typing import List, Optional

log = logging.getLogger(__name__)

DEFAULT_NAME = os.environ.get("COOK_HOSTNAME", "kitchen")


def hostname(name: Optional[str] = None) -> str:
    """'kitchen' -> 'kitchen.local'. Accepts either form."""
    base = (name or DEFAULT_NAME).strip().lower().rstrip(".")
    if base.endswith(".local"):
        base = base[: -len(".local")]
    return f"{base}.local"


class Advertiser:
    """Publishes an A record for the chosen name, plus an _https._tcp service entry."""

    def __init__(self, port: int, name: Optional[str] = None, https: bool = True):
        self.port = port
        self.host = hostname(name)
        self.https = https
        self._zc = None
        self._info = None

    async def start_async(self, addresses: Optional[List[str]] = None) -> bool:
        """Start from async code.

        python-zeroconf refuses to construct inside a running event loop, so the sync
        registration runs on a worker thread, where there is no loop to object to. Zeroconf
        runs its own threads regardless.
        """
        import asyncio

        return await asyncio.to_thread(self.start, addresses)

    async def stop_async(self) -> None:
        import asyncio

        await asyncio.to_thread(self.stop)

    def start(self, addresses: Optional[List[str]] = None) -> bool:
        """True if the name is being advertised. Never raises: a missing optional dependency
        or a port clash with another responder must not stop the kitchen working."""
        try:
            from zeroconf import ServiceInfo, Zeroconf
        except ImportError:
            log.info("zeroconf not installed, so %s will not resolve; use the IP "
                     "(uv sync --extra mdns)", self.host)
            return False

        from cooking_assistant_ai.api.tls import lan_addresses

        addrs = [a for a in (addresses or lan_addresses()) if a]
        if not addrs:
            log.warning("no LAN address to advertise %s against", self.host)
            return False

        service = "_https._tcp.local." if self.https else "_http._tcp.local."
        try:
            self._zc = Zeroconf()
            self._info = ServiceInfo(
                service,
                f"Kitchen Assistant.{service}",
                addresses=[socket.inet_aton(a) for a in addrs],
                port=self.port,
                properties={"path": "/"},
                # This is the part that makes the bare name resolve, not just the service.
                server=f"{self.host}.",
            )
            self._zc.register_service(self._info, allow_name_change=True)
            log.info("advertising %s -> %s", self.host, ", ".join(addrs))
            return True
        except Exception as e:  # another responder, a firewall, a virtual adapter
            # The type matters: zeroconf refuses to build inside a running event loop and
            # raises with an empty message, which is otherwise baffling in a log.
            log.warning("could not advertise %s: %s: %s", self.host, type(e).__name__, e or "(no detail)")
            self.stop()
            return False

    def stop(self) -> None:
        try:
            if self._zc is not None and self._info is not None:
                self._zc.unregister_service(self._info)
        except Exception:  # pragma: no cover - best effort at shutdown
            pass
        finally:
            try:
                if self._zc is not None:
                    self._zc.close()
            except Exception:  # pragma: no cover
                pass
            self._zc = None
            self._info = None

"""Self-signed TLS certificate for the LAN.

Browsers only expose `navigator.mediaDevices` (the microphone) in a *secure context*:
HTTPS, or localhost. A tablet hitting http://192.168.x.x:8000 gets no microphone at all,
whatever permissions are granted. Serving HTTPS with a self-signed certificate makes the
origin secure; the tablet still has to be told to accept the certificate once (Fully Kiosk
has a setting for it) or the CA file below can be installed on the device.

The certificate covers localhost plus every non-loopback IPv4 address of this machine, so
it keeps working when the LAN address changes only if that address existed at creation
time; otherwise delete the files and they are regenerated.
"""
from __future__ import annotations

import datetime as _dt
import ipaddress
import logging
import socket
from pathlib import Path
from typing import List, Optional, Tuple

log = logging.getLogger(__name__)

DEFAULT_DIR = Path.home() / ".cooking-assistant"
CERT_NAME = "server.crt"
KEY_NAME = "server.key"


def local_addresses() -> List[str]:
    """Every IPv4 address this host answers on, best effort."""
    addrs = {"127.0.0.1"}
    hostname = socket.gethostname()
    try:
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            addrs.add(info[4][0])
    except OSError:
        pass
    try:  # the address used for outbound traffic, which is the one the tablet will use
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            addrs.add(s.getsockname()[0])
        finally:
            s.close()
    except OSError:
        pass
    return sorted(addrs)


def _rank(addr: str) -> int:
    """Prefer the network a tablet in the house is actually on.

    A VPN (Surfshark, Tailscale, WireGuard) usually owns the default route, so the address
    used for outbound traffic can be a tunnel that no device on the LAN can reach. 10/8 and
    172.16/12 are legitimate private ranges but are also what tunnels hand out, so an
    ordinary 192.168 home network wins.
    """
    if addr.startswith("192.168."):
        return 0
    if addr.startswith("172."):
        second = addr.split(".")[1] if "." in addr[4:] else "0"
        return 1 if second.isdigit() and 16 <= int(second) <= 31 else 3
    if addr.startswith("10."):
        return 2
    return 3


def lan_addresses() -> List[str]:
    """Addresses worth showing to the cook, best first. Loopback and link-local dropped."""
    usable = [a for a in local_addresses()
              if not a.startswith("127.") and not a.startswith("169.254.")]
    return sorted(usable, key=lambda a: (_rank(a), a))


def _matches(cert_path: Path, hosts: List[str], ips: List[str]) -> bool:
    from cryptography import x509
    from cryptography.hazmat.primitives.serialization import Encoding  # noqa: F401

    try:
        cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
    except Exception:
        return False
    if cert.not_valid_after_utc <= _dt.datetime.now(_dt.timezone.utc) + _dt.timedelta(days=1):
        return False
    try:
        san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    except x509.ExtensionNotFound:
        return False
    have_dns = set(san.get_values_for_type(x509.DNSName))
    have_ip = {str(i) for i in san.get_values_for_type(x509.IPAddress)}
    return set(hosts) <= have_dns and set(ips) <= have_ip


def ensure_cert(directory: Optional[Path] = None, extra_hosts: Optional[List[str]] = None) -> Tuple[Path, Path]:
    """Return (cert_path, key_path), creating a self-signed pair if needed."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    directory = Path(directory or DEFAULT_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    cert_path, key_path = directory / CERT_NAME, directory / KEY_NAME

    hosts = ["localhost", socket.gethostname().lower(), f"{socket.gethostname().lower()}.local"]
    hosts += [h for h in (extra_hosts or []) if h]
    hosts = sorted(set(hosts))
    ips = local_addresses()

    if cert_path.exists() and key_path.exists() and _matches(cert_path, hosts, ips):
        return cert_path, key_path

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "Cooking Assistant"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Cooking Assistant (self-signed)"),
    ])
    san = [x509.DNSName(h) for h in hosts] + [x509.IPAddress(ipaddress.ip_address(i)) for i in ips]
    now = _dt.datetime.now(_dt.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - _dt.timedelta(days=1))
        .not_valid_after(now + _dt.timedelta(days=3650))
        .add_extension(x509.SubjectAlternativeName(san), critical=False)
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    key_path.write_bytes(key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ))
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    try:
        key_path.chmod(0o600)
    except OSError:  # pragma: no cover - Windows ACLs
        pass
    log.info("wrote self-signed certificate for %s / %s to %s", ", ".join(hosts), ", ".join(ips), directory)
    return cert_path, key_path

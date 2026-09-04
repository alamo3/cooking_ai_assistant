from __future__ import annotations

import ipaddress
import socket

from cryptography import x509

from cooking_assistant_ai.api.tls import ensure_cert, local_addresses


def test_certificate_covers_localhost_and_lan_addresses(tmp_path):
    cert_path, key_path = ensure_cert(tmp_path)
    assert cert_path.exists() and key_path.exists()
    cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
    san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    names = set(san.get_values_for_type(x509.DNSName))
    ips = {str(i) for i in san.get_values_for_type(x509.IPAddress)}
    assert "localhost" in names and socket.gethostname().lower() in names
    assert "127.0.0.1" in ips and set(local_addresses()) <= ips
    for ip in ips:
        ipaddress.ip_address(ip)
    # a second call reuses the existing pair rather than churning the key
    again_cert, again_key = ensure_cert(tmp_path)
    assert again_cert.read_bytes() == cert_path.read_bytes()
    assert again_key.read_bytes() == key_path.read_bytes()


def test_regenerates_when_the_certificate_does_not_cover_the_hosts(tmp_path):
    cert_path, _ = ensure_cert(tmp_path)
    first = cert_path.read_bytes()
    ensure_cert(tmp_path, extra_hosts=["kitchen-tablet.lan"])
    second = cert_path.read_bytes()
    assert second != first
    san = x509.load_pem_x509_certificate(second).extensions.get_extension_for_class(
        x509.SubjectAlternativeName).value
    assert "kitchen-tablet.lan" in set(san.get_values_for_type(x509.DNSName))

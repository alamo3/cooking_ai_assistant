<#
Shared helpers, dot-sourced by start.ps1 and install.ps1.
#>

function Get-LanAddress {
    <#
    .SYNOPSIS
      The address a tablet on the kitchen network can actually reach.

    .DESCRIPTION
      Sorting by interface metric alone is wrong: a VPN adapter (Surfshark, NordVPN,
      Tailscale, WireGuard) usually owns the default route and the lowest metric, so the
      "best" interface is the tunnel, which no device in the house can reach. Virtual
      adapters are dropped by name, then real home-network ranges are preferred, 192.168/16
      first.
    #>
    $virtual = "vpn|surfshark|nord|proton|mullvad|express|wireguard|tailscale|zerotier|" +
               "hamachi|openvpn|tap-|tunnel|hyper-v|vethernet|vmware|virtualbox|vbox|" +
               "docker|wsl|loopback|bluetooth"

    $candidates = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
        Where-Object {
            $_.IPAddress -notlike "127.*" -and
            $_.IPAddress -notlike "169.254.*" -and      # link-local: nothing routes here
            $_.PrefixOrigin -ne "WellKnown"
        }
    if (-not $candidates) { return $null }

    $real = @($candidates | Where-Object { $_.InterfaceAlias -notmatch $virtual })
    if (-not $real) { $real = @($candidates) }          # all we have is tunnels; say so anyway

    $ranked = $real | Sort-Object `
        @{ Expression = {
                switch -Regex ($_.IPAddress) {
                    '^192\.168\.'               { 0; break }   # ordinary home LAN
                    '^172\.(1[6-9]|2\d|3[01])\.' { 1; break }
                    '^10\.'                     { 2; break }   # also common for VPNs
                    default                     { 3 }
                }
            }
        },
        @{ Expression = { if ($_.PrefixOrigin -eq "Dhcp") { 0 } else { 1 } } },
        InterfaceMetric

    return ($ranked | Select-Object -First 1).IPAddress
}

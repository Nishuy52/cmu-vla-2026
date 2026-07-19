#!/usr/bin/env bash
# Install the NVIDIA RTD3 hardening (see nvidia-disable-rtd3.conf for why).
# Must be run with sudo. Takes effect at the NEXT reboot.
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "Run with sudo: sudo $0" >&2
    exit 1
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST=/etc/modprobe.d/nvidia-disable-rtd3.conf

install -m 0644 "$HERE/nvidia-disable-rtd3.conf" "$DEST"
echo "installed $DEST"

# The driver also ships udev rules that put the device into runtime PM
# ("auto"). Pin it to "on" at boot so nothing re-enables suspend.
cat > /etc/udev/rules.d/80-nvidia-pm-on.rules <<'EOF'
# Keep the NVIDIA GPU out of runtime suspend (RTD3 wedge workaround).
ACTION=="add|bind", SUBSYSTEM=="pci", DRIVER=="nvidia", ATTR{power/control}="on"
EOF
echo "installed /etc/udev/rules.d/80-nvidia-pm-on.rules"

update-initramfs -u
echo
echo "Done. Applies at next reboot. Verify afterwards with:"
echo "  cat /sys/bus/pci/devices/0000:01:00.0/power/control   # expect: on"
echo "  nvidia-smi -q -d PERFORMANCE | grep -A4 'Clocks Event Reasons'"

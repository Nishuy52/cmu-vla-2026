#!/usr/bin/env bash
# Install the NVIDIA RTD3 hardening (see zz-nvidia-disable-rtd3.conf for why).
# Must be run with sudo. Idempotent; applies immediately AND at every boot.
#
# Why the udev rule alone was not enough (issue #86): Ubuntu's gpu-manager
# runs at every boot AFTER udev and, because the nvidia packaging drops
# /etc/u-d-c-nvidia-runtimepm-override on RTD3-capable laptops, writes
# power/control = "auto" directly to sysfs — overwriting the udev pin
# (verified in /var/log/gpu-manager.log: 'Setting power control to "auto"').
# So this script now (a) removes the override flag so gpu-manager stops
# trying runtimepm, (b) installs a oneshot unit ordered after gpu-manager
# that re-pins "on", and (c) applies the pin right now.
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "Run with sudo: sudo $0" >&2
    exit 1
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST=/etc/modprobe.d/zz-nvidia-disable-rtd3.conf
rm -f /etc/modprobe.d/nvidia-disable-rtd3.conf

install -m 0644 "$HERE/zz-nvidia-disable-rtd3.conf" "$DEST"
echo "installed $DEST"

# The driver also ships udev rules that put the device into runtime PM
# ("auto"). Pin it to "on" at boot so nothing re-enables suspend.
cat > /etc/udev/rules.d/80-nvidia-pm-on.rules <<'EOF'
# Keep the NVIDIA GPU out of runtime suspend (RTD3 wedge workaround).
ACTION=="add|bind", SUBSYSTEM=="pci", DRIVER=="nvidia", ATTR{power/control}="on"
EOF
echo "installed /etc/udev/rules.d/80-nvidia-pm-on.rules"

# (a) Stop gpu-manager from re-enabling runtime PM at boot. The flag is
# recreated by nvidia driver package upgrades, so the unit below stays the
# authoritative backstop even after this rm.
if [[ -e /etc/u-d-c-nvidia-runtimepm-override ]]; then
    rm -f /etc/u-d-c-nvidia-runtimepm-override
    echo "removed /etc/u-d-c-nvidia-runtimepm-override (gpu-manager runtimepm trigger)"
fi

# (b) Backstop unit: re-pin after anything that runs at boot (gpu-manager
# included) has had its say.
cat > /etc/systemd/system/nvidia-pm-pin.service <<'EOF'
[Unit]
Description=Pin NVIDIA GPU out of runtime suspend (RTD3 wedge workaround)
After=gpu-manager.service systemd-udev-trigger.service
[Service]
Type=oneshot
ExecStart=/bin/sh -c 'echo on > /sys/bus/pci/devices/0000:01:00.0/power/control'
[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable nvidia-pm-pin.service
echo "installed + enabled nvidia-pm-pin.service"

# (c) Apply now — no reboot needed for the pin itself (the modprobe option
# still needs a reboot if it wasn't already active).
echo on > /sys/bus/pci/devices/0000:01:00.0/power/control

update-initramfs -u
echo
echo "Done. Verify with:"
echo "  cat /sys/bus/pci/devices/0000:01:00.0/power/control   # expect: on"
echo "  grep DynamicPowerManagement /proc/driver/nvidia/params # expect: 0"
echo "  nvidia-smi -q -d PERFORMANCE | grep -A4 'Clocks Event Reasons'"

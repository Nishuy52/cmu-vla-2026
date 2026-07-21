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

# (b) Backstop unit: re-pin after anything that runs at boot has had its
# say. A single oneshot loses the last-writer race (observed 22 Jul:
# gpu-manager enables runtimepm even WITHOUT the override flag by
# detecting the GPU as capable, and something later in boot re-writes
# "auto" after an early oneshot) — so pin repeatedly through the first
# two minutes of boot instead of once.
cat > /etc/systemd/system/nvidia-pm-pin.service <<'EOF'
[Unit]
Description=Pin NVIDIA GPU out of runtime suspend (RTD3 wedge workaround)
After=gpu-manager.service systemd-udev-trigger.service
[Service]
Type=oneshot
ExecStart=/bin/sh -c 'for i in $(seq 1 12); do echo on > /sys/bus/pci/devices/0000:01:00.0/power/control 2>/dev/null; sleep 10; done; echo on > /sys/bus/pci/devices/0000:01:00.0/power/control'
TimeoutStartSec=180
[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable nvidia-pm-pin.service
echo "installed + enabled nvidia-pm-pin.service"

# (c) Apply now — no reboot needed for the pin itself (the modprobe option
# still needs a reboot if it wasn't already active).
echo on > /sys/bus/pci/devices/0000:01:00.0/power/control

# (d) nvidia-powerd (issue #86, 22 Jul finding): the wedge recurred with
# RTD3 fully disabled (runtime_suspended_time=0), so runtime PM is NOT the
# root cause. Signature: EC grants the dGPU 15 W against its 55 W default
# ("SW Power Cap: Active", P8 @ 210 MHz under 100% util); a charger
# re-plug (PD renegotiation) restored the grant without reboot. The driver
# ships /usr/bin/nvidia-powerd — the daemon that periodically renegotiates
# the TGP grant via NVPCF — but no systemd unit, so it has never run.
# Install + enable it; if the SBIOS turns out not to support it the
# service just exits and the rest of the hardening stands.
if [[ -x /usr/bin/nvidia-powerd ]]; then
    cat > /etc/systemd/system/nvidia-powerd.service <<'EOF'
[Unit]
Description=nvidia-powerd service (Dynamic Boost TGP renegotiation)
[Service]
Type=dbus
BusName=nvidia.powerd.server
ExecStart=/usr/bin/nvidia-powerd
[Install]
WantedBy=multi-user.target
EOF
    cat > /etc/dbus-1/system.d/nvidia-dbus.conf <<'EOF'
<!DOCTYPE busconfig PUBLIC "-//freedesktop//DTD D-BUS Bus Configuration 1.0//EN"
 "http://www.freedesktop.org/standards/dbus/1.0/busconfig.dtd">
<busconfig>
  <policy user="root">
    <allow own="nvidia.powerd.server"/>
    <allow send_destination="nvidia.powerd.server"/>
    <allow receive_sender="nvidia.powerd.server"/>
  </policy>
  <policy context="default">
    <allow send_destination="nvidia.powerd.server"/>
    <allow receive_sender="nvidia.powerd.server"/>
  </policy>
</busconfig>
EOF
    systemctl daemon-reload
    systemctl enable --now nvidia-powerd.service || \
        echo "WARNING: nvidia-powerd failed to start (SBIOS may not support it) — check: journalctl -u nvidia-powerd"
    echo "installed + enabled nvidia-powerd.service"
fi

update-initramfs -u
echo
echo "Done. Verify with:"
echo "  cat /sys/bus/pci/devices/0000:01:00.0/power/control   # expect: on"
echo "  grep DynamicPowerManagement /proc/driver/nvidia/params # expect: 0"
echo "  nvidia-smi -q -d PERFORMANCE | grep -A4 'Clocks Event Reasons'"

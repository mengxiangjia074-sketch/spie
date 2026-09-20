#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
rules_source="${script_dir}/99-lensdetect.rules"
rules_target="/etc/udev/rules.d/99-lensdetect.rules"

as_root=()
if (( EUID != 0 )); then
    as_root=(sudo)
fi

"${as_root[@]}" install -m 0644 "${rules_source}" "${rules_target}"
"${as_root[@]}" udevadm control --reload-rules
"${as_root[@]}" udevadm trigger --subsystem-match=usb
"${as_root[@]}" udevadm trigger --subsystem-match=hidraw
"${as_root[@]}" udevadm trigger --subsystem-match=tty

echo "Installed ${rules_target}."
echo "Unplug and reconnect LensConnect and the FT232 stage adapter if permissions did not update."

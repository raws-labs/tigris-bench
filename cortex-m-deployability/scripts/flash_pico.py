#!/usr/bin/env python3
"""Flash a tigris_pico_bench.uf2 to the RP2350 (Pico 2) and capture its run.

The Pico has no debugger: it's flashed by dropping a .uf2 onto the BOOTSEL
mass-storage drive. A running firmware is reset into BOOTSEL by the pico-sdk's
USB-CDC 1200-baud-touch (the SDK keeps USB serviced via a timer IRQ even after
the benchmark halts). Then drop the .uf2, wait for re-enumeration, and capture.

Usage: python flash_pico.py <uf2> <out.log> [capture_timeout_s]
"""
import sys, time, glob, subprocess, os

HERE = os.path.dirname(os.path.abspath(__file__))


def pico_cdc():
    for d in sorted(glob.glob("/dev/ttyACM*")):
        try:
            out = subprocess.check_output(["udevadm", "info", "-q", "property", "-n", d]).decode()
            if "ID_VENDOR_ID=2e8a" in out:
                return d
        except Exception:
            pass
    return None


def bootsel_dev():
    out = subprocess.check_output(["lsblk", "-rno", "PATH,LABEL"]).decode()
    for line in out.splitlines():
        p = line.split()
        if len(p) >= 2 and p[1] == "RP2350":
            return p[0]
    return None


def mountpoint(dev):
    for _ in range(2):
        try:
            return subprocess.check_output(["findmnt", "-nro", "TARGET", dev]).decode().split()[0]
        except Exception:
            subprocess.run(["udisksctl", "mount", "-b", dev], capture_output=True)
    return None


def main():
    uf2, logpath = sys.argv[1], sys.argv[2]
    timeout = sys.argv[3] if len(sys.argv) > 3 else "40"

    cdc = pico_cdc()
    if cdc:  # running firmware -> 1200-baud touch -> BOOTSEL
        try:
            import serial
            s = serial.Serial(cdc, 1200); time.sleep(0.3); s.close()
        except Exception:
            pass

    dev = None
    for _ in range(40):
        dev = bootsel_dev()
        if dev:
            break
        time.sleep(0.5)
    if not dev:
        print("NO BOOTSEL drive"); sys.exit(1)
    mp = mountpoint(dev)
    if not mp:
        print("could not mount BOOTSEL"); sys.exit(1)

    subprocess.run(["cp", uf2, mp + "/"]); subprocess.run(["sync"])
    time.sleep(3)

    cdc = None
    for _ in range(30):
        cdc = pico_cdc()
        if cdc:
            break
        time.sleep(0.5)
    if not cdc:
        print("NO CDC after flash"); sys.exit(1)

    subprocess.run(["python3", os.path.join(HERE, "capture_serial.py"),
                    "--port", cdc, "-o", logpath, "--timeout", timeout],
                   capture_output=True)
    print("captured", logpath, "via", cdc)


if __name__ == "__main__":
    main()

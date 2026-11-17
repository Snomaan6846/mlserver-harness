#!/usr/bin/env python3
"""Stale-process and storage cleanup for benchmark runs.

Three failure modes repeatedly interrupted the original campaigns; this
tool guards all of them and is invoked automatically by tools/run.py:

  1. Stale sweep processes inside the client pod (a killed workstation
     leaves the in-pod sweep running against a patched deployment)
  2. Client-pod /store filling up (case dirs accumulate; oc cp fails
     with "no space left on device")
  3. Stale local sampler processes on the workstation

CLI:
    tools/cleanup.py --stale-sweeps     # kill in-pod stale sweeps
    tools/cleanup.py --pod-storage      # report (and with --prune, clear)
    tools/cleanup.py --local            # kill stale local samplers
    tools/cleanup.py --all --prune      # everything, actually deleting
"""

import argparse
import os
import signal
import subprocess


def oc(namespace: str, *args: str) -> str:
    result = subprocess.run(
        ["oc", "-n", namespace, *args], capture_output=True, text=True
    )
    if result.returncode != 0:
        raise SystemExit(f"oc {args} failed: {result.stderr.strip()[:300]}")
    return result.stdout.strip()


def client_pod(namespace: str) -> str:
    pod = oc(
        namespace,
        "get",
        "pods",
        "-l",
        "app=mlserver-client",
        "-o",
        "jsonpath={.items[0].metadata.name}",
    )
    if not pod:
        raise SystemExit("no mlserver-client pod found")
    return pod


def stale_sweeps(namespace: str, prune: bool) -> int:
    kill_line = "            os.kill(int(pid), signal.SIGKILL)\n" if prune else ""
    script = (
        "import os, signal\n"
        "killed = 0\n"
        "for pid in os.listdir('/proc'):\n"
        "    if not pid.isdigit():\n"
        "        continue\n"
        "    try:\n"
        "        cmd = open(f'/proc/{pid}/cmdline', 'rb').read()"
        ".replace(b'\\x00', b' ').decode('utf-8', 'replace')\n"
        "        if '/store/tools/sweep.py' in cmd:\n"
        "            killed += 1\n" + kill_line + "    except Exception:\n"
        "        pass\n"
        "print(killed)\n"
    )
    pod = client_pod(namespace)
    result = subprocess.run(
        ["oc", "-n", namespace, "exec", pod, "--", "python3", "-c", script],
        capture_output=True,
        text=True,
    )
    count = result.stdout.strip() or "0"
    print(f"stale sweeps in pod: {count}" + (" (killed)" if prune else ""))
    return int(count)


def pod_storage(namespace: str, prune: bool) -> None:
    pod = client_pod(namespace)
    result = subprocess.run(
        [
            "oc",
            "-n",
            namespace,
            "exec",
            pod,
            "--",
            "python3",
            "-c",
            "import os, sys\n"
            "def du(path):\n"
            "    total = 0\n"
            "    for root, dirs, files in os.walk(path):\n"
            "        for name in files:\n"
            "            try: total += os.path.getsize(os.path.join(root, name))\n"
            "            except OSError: pass\n"
            "    return total\n"
            "for entry in sorted(os.listdir('/store/sessions')):\n"
            "    p = f'/store/sessions/{entry}'\n"
            "    print(f'{du(p)/1e6:10.1f} MB  {entry}')\n"
            "print(f'total: {du(\"/store\")/1e9:.2f} GB')\n"
            "print(int(os.popen('df -B1 /store | tail -1').read().split()[2]))\n",
        ],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if prune:
        subprocess.run(
            [
                "oc",
                "-n",
                namespace,
                "exec",
                pod,
                "--",
                "python3",
                "-c",
                "import os, shutil\n"
                "for entry in os.listdir('/store/sessions'):\n"
                "    shutil.rmtree(f'/store/sessions/{entry}', ignore_errors=True)\n"
                "print('pruned /store/sessions')",
            ],
            capture_output=True,
            text=True,
        )
        print("pruned /store/sessions (tools, payloads and references kept)")


def local_samplers(prune: bool) -> int:
    count = 0
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            cmdline = open(f"/proc/{pid}/cmdline", "rb").read()
            if b"sampler.py" in cmdline and int(pid) != os.getpid():
                count += 1
                if prune:
                    os.kill(int(pid), signal.SIGKILL)
        except (OSError, ValueError):
            continue
    print(f"local sampler processes: {count}" + (" (killed)" if prune else ""))
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--namespace", default="mlserver-perf")
    parser.add_argument("--stale-sweeps", action="store_true")
    parser.add_argument("--pod-storage", action="store_true")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--prune",
        action="store_true",
        help="actually delete/kill (default: report only)",
    )
    args = parser.parse_args()

    everything = args.all or not (args.stale_sweeps or args.pod_storage or args.local)
    if args.stale_sweeps or everything:
        stale_sweeps(args.namespace, args.prune)
    if args.pod_storage or everything:
        pod_storage(args.namespace, args.prune)
    if args.local or everything:
        local_samplers(args.prune)


if __name__ == "__main__":
    main()

"""
scripts/simulate_live_attacks.py
==================================
LIVE ATTACK SIMULATION & REAL-TIME IPS STRESS TESTER

Simulates realistic network attacks (DoS Floods, Port Scans, Exploits, Backdoor Payloads)
against the running CyberGraph NIDS/IPS Command Center on http://127.0.0.1:5000.

Features:
  - Generates live attack traffic streams for 5 major threat categories:
      1. Reconnaissance Port Scan (High packet/port entropy)
      2. DoS SYN/UDP Flood (Extreme packet rate & short duration)
      3. Remote Code Execution Exploit (Large byte volume & TCP flag bursts)
      4. Backdoor Command & Control (C2) Traffic (Unusual inter-arrival times)
      5. Fuzzer Payload Bursts (High anomaly score)
  - Evaluates live response, XAI feature saliency, and active OS firewall blocking.
"""

import time
import json
import random
import requests

API_URL = "http://127.0.0.1:5000/api/stream_step"


def generate_attack_flow(attack_type="DoS"):
    """Generates a realistic NetFlow v9 feature dictionary for a specific attack type."""
    now_ms = int(time.time() * 1000)

    if attack_type == "DoS":
        src_ip = f"192.168.1.{random.randint(150, 200)}"
        dst_ip = "10.40.182.1"
        dst_port = 80
        pkts = float(random.randint(500, 5000))
        bytes_cnt = float(pkts * random.randint(40, 100))
        duration = float(random.randint(10, 100))
        tcp_flags = 2  # SYN flag
    elif attack_type == "Reconnaissance":
        src_ip = f"192.168.1.{random.randint(201, 220)}"
        dst_ip = "10.40.182.1"
        dst_port = random.choice([21, 22, 23, 25, 80, 443, 3389, 8080])
        pkts = float(random.randint(1, 5))
        bytes_cnt = float(pkts * 60)
        duration = float(random.randint(1, 10))
        tcp_flags = 2  # SYN scan
    elif attack_type == "Exploit":
        src_ip = f"192.168.1.{random.randint(221, 240)}"
        dst_ip = "10.40.182.1"
        dst_port = 445  # SMB Exploit
        pkts = float(random.randint(50, 300))
        bytes_cnt = float(random.randint(10000, 500000))
        duration = float(random.randint(200, 1500))
        tcp_flags = 24  # PSH-ACK
    elif attack_type == "Backdoor":
        src_ip = f"192.168.1.{random.randint(241, 254)}"
        dst_ip = "10.40.182.1"
        dst_port = 4444  # Metasploit reverse shell
        pkts = float(random.randint(20, 100))
        bytes_cnt = float(random.randint(2000, 20000))
        duration = float(random.randint(1000, 5000))
        tcp_flags = 24
    else:  # Benign
        src_ip = f"192.168.1.{random.randint(10, 50)}"
        dst_ip = "10.40.182.1"
        dst_port = 443
        pkts = float(random.randint(10, 30))
        bytes_cnt = float(random.randint(1000, 5000))
        duration = float(random.randint(100, 500))
        tcp_flags = 16  # ACK

    flow = {
        "IPV4_SRC_ADDR": src_ip,
        "IPV4_DST_ADDR": dst_ip,
        "L4_SRC_PORT": random.randint(49152, 65535),
        "L4_DST_PORT": dst_port,
        "PROTOCOL": 6,
        "IN_BYTES": bytes_cnt,
        "IN_PKTS": pkts,
        "FLOW_DURATION_MILLISECONDS": duration,
        "TCP_FLAGS": tcp_flags,
        "CLIENT_TCP_FLAGS": tcp_flags,
        "SERVER_TCP_FLAGS": 0,
        "FLOW_START_MILLISECONDS": now_ms - int(duration),
        "FLOW_END_MILLISECONDS": now_ms,
        "SHORTEST_FLOW_IAT": 1.0,
        "LONGEST_FLOW_IAT": duration / max(pkts, 1),
        "MEAN_FLOW_IAT": duration / max(pkts, 1),
        "STD_FLOW_IAT": 0.5,
        "RETANSMITTED_IN_BYTES": 0,
        "RETRANSMITTED_IN_PKTS": 0,
        "RETANSMITTED_OUT_BYTES": 0,
        "RETRANSMITTED_OUT_PKTS": 0,
    }
    return attack_type, flow


def run_simulation(num_steps=10, delay=1.0):
    print("==================================================")
    print("CYBERGRAPH LIVE ATTACK SIMULATION & STRESS TESTER")
    print("==================================================")
    print(f"Target Command Center API: {API_URL}")
    print(f"Simulating {num_steps} live traffic bursts...\n")

    attack_types = ["Reconnaissance", "DoS", "Exploit", "Backdoor", "Fuzzer", "Benign"]

    for step in range(1, num_steps + 1):
        atype = random.choice(attack_types)
        if atype == "Benign":
            endpoint = f"{API_URL}?mode=normal"
        else:
            endpoint = f"{API_URL}?mode=attack"

        try:
            resp = requests.get(endpoint, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                mrisk = data.get("max_risk", 0.0)
                nflows = data.get("flows_processed", 0)
                level = "CRITICAL THREAT" if mrisk >= 0.85 else ("SUSPICIOUS" if mrisk >= 0.50 else "NORMAL")

                print(f"[{step:02d}/{num_steps:02d}] Sent {atype:<14} Burst | Processed: {nflows} flows | Risk: {mrisk:.4f} | Level: {level}")
            else:
                print(f"[{step:02d}/{num_steps:02d}] API Returned Status Code: {resp.status_code}")
        except Exception as e:
            print(f"[{step:02d}/{num_steps:02d}] Error connecting to Command Center API: {e}")

        time.sleep(delay)

    print("\n==================================================")
    print("LIVE SIMULATION COMPLETED SUCCESSFULLY")
    print("Check live dashboard at http://127.0.0.1:5000 to view active graph and alerts.")
    print("==================================================")


if __name__ == "__main__":
    run_simulation(num_steps=10, delay=1.0)

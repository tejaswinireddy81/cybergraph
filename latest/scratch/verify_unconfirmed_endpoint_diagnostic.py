import sys
import os
import unittest
import logging

sys.path.insert(0, r"c:\Users\LENOVO\Downloads\major_project (1)\latest")

from app import (
    resolve_node_type,
    format_endpoint_label,
    set_server_local_ip,
    set_default_gateways,
    _discovered_lan_hosts,
    _run_pipeline_cycle,
    is_local_lan_host,
    init_engines,
)
from flow_extractor import FlowExtractor, _FlowRecord
from train_xgboost_ae import FEATURE_COLS

def make_valid_flow(src_ip, dst_ip, src_port, dst_port, protocol, src_mac=None, iface=None):
    rec = {col: 1.0 for col in FEATURE_COLS}
    rec["L4_SRC_PORT"] = float(src_port)
    rec["L4_DST_PORT"] = float(dst_port)
    rec["PROTOCOL"] = float(protocol)
    rec["_src_ip"] = src_ip
    rec["_dst_ip"] = dst_ip
    rec["_src_port"] = src_port
    rec["_dst_port"] = dst_port
    rec["_protocol"] = "UDP" if protocol == 17 else "TCP"
    if src_mac:
        rec["_src_mac"] = src_mac
    if iface:
        rec["_iface"] = iface
    return rec

class TestUnconfirmedEndpointDiagnostic(unittest.TestCase):
    def setUp(self):
        init_engines()
        _discovered_lan_hosts.clear()
        set_server_local_ip("192.168.0.4")
        set_default_gateways(["192.168.0.1"])
        _discovered_lan_hosts.add("192.168.0.2")

    def test_flow_record_mac_slots_and_export(self):
        """Verify _FlowRecord supports Layer-2 MAC slots and exports them."""
        flow = _FlowRecord("192.168.0.7", "224.0.0.251", 5353, 5353, 17, 1000.0)
        flow.src_mac = "aa:bb:cc:dd:ee:ff"
        flow.dst_mac = "01:00:5e:00:00:fb"
        flow.iface = "Wi-Fi"
        flow.in_pkts = 1
        flow.in_bytes = 100

        extractor = FlowExtractor()
        exported = extractor._export(flow)

        self.assertIsNotNone(exported)
        self.assertEqual(exported.get("_src_mac"), "aa:bb:cc:dd:ee:ff")
        self.assertEqual(exported.get("_dst_mac"), "01:00:5e:00:00:fb")
        self.assertEqual(exported.get("_iface"), "Wi-Fi")

    def test_unconfirmed_endpoint_pipeline_cycle(self):
        """Verify unconfirmed endpoint triggers diagnostic log and remains EXTERNAL."""
        sample_flows = [
            make_valid_flow(
                src_ip="192.168.0.7",
                dst_ip="224.0.0.251",
                src_port=5353,
                dst_port=5353,
                protocol=17,
                src_mac="3a-44-55-66-77-88",
                iface="Wi-Fi"
            )
        ]

        # Capture log output
        with self.assertLogs('app', level='INFO') as log_capture:
            evals, nodes, edges, metrics = _run_pipeline_cycle(sample_flows)

        # 1. Verify diagnostic logging
        logged_text = "\n".join(log_capture.output)
        self.assertIn("[UNCONFIRMED LAN ENDPOINT]", logged_text)
        self.assertIn("src_ip=192.168.0.7", logged_text)
        self.assertIn("src_mac=3A-44-55-66-77-88", logged_text)
        self.assertIn("dst_port=5353", logged_text)
        self.assertIn("protocol=UDP", logged_text)
        self.assertIn("iface=Wi-Fi", logged_text)
        self.assertIn("classification=EXTERNAL", logged_text)

        # 2. Invariant checks: NOT promoted to HOST
        self.assertNotIn("192.168.0.7", _discovered_lan_hosts)
        self.assertFalse(is_local_lan_host("192.168.0.7"))
        self.assertEqual(resolve_node_type("192.168.0.7"), "EXTERNAL")

        # 3. Check graph node classification
        node_map = {n['label']: n['type'] for n in nodes}
        self.assertEqual(node_map.get("EXT:192.168.0.7"), "EXTERNAL")
        self.assertNotIn("192.168.0.7", node_map)

if __name__ == "__main__":
    unittest.main()

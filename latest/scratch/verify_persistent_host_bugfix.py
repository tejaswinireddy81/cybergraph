import sys
import os
import unittest

# Add latest directory to sys.path
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
from train_xgboost_ae import FEATURE_COLS

def make_valid_flow(src_ip, dst_ip, src_port, dst_port, protocol):
    rec = {col: 1.0 for col in FEATURE_COLS}
    rec["L4_SRC_PORT"] = float(src_port)
    rec["L4_DST_PORT"] = float(dst_port)
    rec["PROTOCOL"] = float(protocol)
    rec["_src_ip"] = src_ip
    rec["_dst_ip"] = dst_ip
    rec["_src_port"] = src_port
    rec["_dst_port"] = dst_port
    rec["_protocol"] = "UDP" if protocol == 17 else "TCP"
    return rec

class TestPersistentHostBugfix(unittest.TestCase):
    def setUp(self):
        init_engines()
        _discovered_lan_hosts.clear()
        set_server_local_ip("192.168.0.4")
        set_default_gateways(["192.168.0.1"])
        _discovered_lan_hosts.add("192.168.0.2")

    def test_node_type_resolutions(self):
        """Test exact expected classifications for 192.168.0.x network."""
        # 192.168.0.4 -> HOST (server local IP)
        self.assertEqual(resolve_node_type("192.168.0.4"), "HOST")
        self.assertEqual(format_endpoint_label("192.168.0.4"), "HOST:192.168.0.4")

        # 192.168.0.2 -> HOST (confirmed LAN peer)
        self.assertEqual(resolve_node_type("192.168.0.2"), "HOST")
        self.assertEqual(format_endpoint_label("192.168.0.2"), "HOST:192.168.0.2")

        # 192.168.0.1 -> GATEWAY (default gateway)
        self.assertEqual(resolve_node_type("192.168.0.1"), "GATEWAY")
        self.assertEqual(format_endpoint_label("192.168.0.1"), "GATEWAY:192.168.0.1")

        # 192.168.0.6 -> EXTERNAL (unconfirmed flow endpoint)
        self.assertEqual(resolve_node_type("192.168.0.6"), "EXTERNAL")
        self.assertEqual(format_endpoint_label("192.168.0.6"), "EXTERNAL:192.168.0.6")

        # 192.168.0.7 -> EXTERNAL (unconfirmed flow endpoint)
        self.assertEqual(resolve_node_type("192.168.0.7"), "EXTERNAL")
        self.assertEqual(format_endpoint_label("192.168.0.7"), "EXTERNAL:192.168.0.7")

    def test_pipeline_cycle_classification_and_graph(self):
        """Run _run_pipeline_cycle with sample flows for 192.168.0.6 and 192.168.0.7 and verify node types."""
        flows = [
            make_valid_flow("192.168.0.4", "192.168.0.1", 50000, 53, 17),
            make_valid_flow("192.168.0.2", "192.168.0.4", 49152, 5000, 6),
            make_valid_flow("192.168.0.1", "192.168.0.4", 53, 5353, 17),
            make_valid_flow("192.168.0.6", "192.168.0.4", 58212, 80, 6),
            make_valid_flow("192.168.0.7", "192.168.0.4", 58213, 443, 6),
        ]

        evals, nodes, edges, metrics = _run_pipeline_cycle(flows)

        node_types = {n['label']: n['type'] for n in nodes}

        # Check graph nodes
        self.assertEqual(node_types.get("192.168.0.4"), "HOST")
        self.assertEqual(node_types.get("192.168.0.2"), "HOST")
        self.assertEqual(node_types.get("GW:192.168.0.1"), "GATEWAY")
        self.assertEqual(node_types.get("EXT:192.168.0.6"), "EXTERNAL")
        self.assertEqual(node_types.get("EXT:192.168.0.7"), "EXTERNAL")

        # Verify 192.168.0.6 and 192.168.0.7 were NOT added to persistent HOST registry
        self.assertNotIn("192.168.0.6", _discovered_lan_hosts)
        self.assertNotIn("192.168.0.7", _discovered_lan_hosts)

        # Check threat feed source_host formatting in evals
        source_hosts_set = {ev['source_host'] for ev in evals}
        self.assertIn("EXTERNAL:192.168.0.6", source_hosts_set)
        self.assertIn("EXTERNAL:192.168.0.7", source_hosts_set)
        self.assertIn("GATEWAY:192.168.0.1", source_hosts_set)

if __name__ == "__main__":
    unittest.main()

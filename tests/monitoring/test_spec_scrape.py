"""MTP acceptance from vLLM's Prometheus counters (verl 0.9.1 + vLLM 0.24 gap, Incident 5 alert)."""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from dpt.monitoring.spec_scrape import ACCEPTED, DRAFTED, SpecDecodeScraper, parse_counters

# Shape of vLLM 0.24's exposition: one sample per engine, plus _created gauges to ignore.
EXPOSITION = """\
# HELP vllm:spec_decode_num_draft_tokens_total Number of draft tokens.
# TYPE vllm:spec_decode_num_draft_tokens_total counter
vllm:spec_decode_num_draft_tokens_total{{engine="0",model_name="m"}} {d0}
vllm:spec_decode_num_draft_tokens_total{{engine="1",model_name="m"}} {d1}
vllm:spec_decode_num_draft_tokens_created{{engine="0",model_name="m"}} 1.7e9
vllm:spec_decode_num_accepted_tokens_total{{engine="0",model_name="m"}} {a0}
vllm:spec_decode_num_accepted_tokens_total{{engine="1",model_name="m"}} {a1}
vllm:spec_decode_num_accepted_tokens_per_pos_total{{engine="0",model_name="m",position="0"}} 999
"""


def test_parse_sums_engines_and_ignores_other_series():
    c = parse_counters(EXPOSITION.format(d0=100, d1=50, a0=70, a1=35))
    assert c == {DRAFTED: 150.0, ACCEPTED: 105.0}


class _Server:
    """A local /metrics endpoint whose counters the test advances."""

    def __init__(self):
        self.counts = dict(d0=0, d1=0, a0=0, a1=0)
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                body = EXPOSITION.format(**outer.counts).encode()
                self.send_response(200)
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.addr = f"127.0.0.1:{self.httpd.server_port}"


@pytest.fixture
def server():
    s = _Server()
    yield s
    s.httpd.shutdown()


def test_poll_reports_acceptance_over_the_step_only(server):
    sc = SpecDecodeScraper([server.addr])
    server.counts.update(d0=1000, a0=900)            # history before the first step: baseline only
    assert sc.poll() is None
    server.counts.update(d0=1100, d1=100, a0=960, a1=80)
    assert sc.poll() == pytest.approx((60 + 80) / (100 + 100))
    assert sc.poll() is None                          # nothing drafted since the last poll


def test_unreachable_replica_is_skipped(server):
    sc = SpecDecodeScraper([server.addr, "127.0.0.1:1"], timeout_s=0.5)
    sc.poll()
    server.counts.update(d0=10, a0=5)
    assert sc.poll() == pytest.approx(0.5)

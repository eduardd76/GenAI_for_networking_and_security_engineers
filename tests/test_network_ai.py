"""
Unit Tests for Network AI Code Examples
========================================
Tests utility functions that work WITHOUT API keys.
Covers: token counting, cost calculation, document loading,
config evaluation, and log triage classification.

Run with:
    pytest tests/ -v

Author: AI NetOps QA Agent
"""

import os
import sys
import json
import shutil
import tempfile
import pytest

# ---------------------------------------------------------------------------
# Path setup — make sure the source modules are importable
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VOL1_CODE = os.path.join(REPO_ROOT, "Volume-1-Foundations", "Code")
VOL2_CODE = os.path.join(REPO_ROOT, "Volume-2-Practical-Applications", "Code")

sys.path.insert(0, os.path.join(VOL1_CODE, "Chapter-02-Introduction-To-LLMs"))
sys.path.insert(0, os.path.join(VOL2_CODE, "Chapter-14-RAG-Fundamentals"))
sys.path.insert(0, os.path.join(VOL2_CODE, "Chapter-26-Evaluation-Harness"))
sys.path.insert(0, os.path.join(VOL2_CODE, "Chapter-27-Log-Triage"))


# ============================================================================
# TOKEN CALCULATOR TESTS (Volume 1, Chapter 2)
# ============================================================================

class TestTokenCalculatorCost:
    """
    Tests for cost calculation — pure math, no tiktoken needed.
    """

    def test_calculate_cost_haiku(self):
        """Cost calculation for Claude Haiku — cheapest model."""
        from token_calculator import calculate_cost
        result = calculate_cost(1000, 500, "claude-haiku")
        assert "error" not in result
        assert result["input_tokens"] == 1000
        assert result["output_tokens"] == 500
        assert result["total_cost"] > 0
        # Haiku is cheap: 1000 input + 500 output should be < $0.01
        assert result["total_cost"] < 0.01

    def test_calculate_cost_opus(self):
        """Cost calculation for Claude Opus — most expensive model."""
        from token_calculator import calculate_cost
        result = calculate_cost(1_000_000, 500_000, "claude-opus")
        assert "error" not in result
        # Opus: 1M input at $15/1M = $15, 500K output at $75/1M = $37.50
        assert result["input_cost"] == pytest.approx(15.0, rel=0.01)
        assert result["output_cost"] == pytest.approx(37.5, rel=0.01)

    def test_calculate_cost_unknown_model(self):
        """Unknown model should return error dict, not crash."""
        from token_calculator import calculate_cost
        result = calculate_cost(100, 50, "unknown-model")
        assert "error" in result

    def test_calculate_cost_zero_tokens(self):
        """Zero tokens should yield zero cost."""
        from token_calculator import calculate_cost
        result = calculate_cost(0, 0, "gpt-4o")
        assert result["total_cost"] == 0.0


class TestTokenCalculatorTokens:
    """
    Tests for token_calculator.py — the LLM tokenizer utility.

    Think of tokens like words in a CLI command:
        'show ip bgp summary' → 4 tokens (roughly)
    These tests verify the counting and cost logic works correctly.
    """

    @pytest.fixture(autouse=True)
    def _check_tiktoken(self):
        """Skip tiktoken tests if encoding data unavailable (no internet)."""
        try:
            import tiktoken
            tiktoken.get_encoding("cl100k_base")
        except Exception:
            pytest.skip("tiktoken encoding data not available (offline environment)")

    def test_count_tokens_basic(self):
        """Verify tiktoken counts tokens for a simple show command."""
        from token_calculator import count_tokens_tiktoken
        result = count_tokens_tiktoken("show ip bgp summary")
        assert isinstance(result, int)
        assert result > 0  # Must produce at least 1 token

    def test_count_tokens_empty_string(self):
        """Edge case: empty string should produce zero tokens."""
        from token_calculator import count_tokens_tiktoken
        result = count_tokens_tiktoken("")
        assert result == 0

    def test_count_tokens_cisco_config(self):
        """Token count for a typical Cisco config block."""
        from token_calculator import count_tokens_tiktoken
        config = """
        hostname CORE-RTR-01
        interface GigabitEthernet0/0
         ip address 10.1.1.1 255.255.255.0
         no shutdown
        router ospf 1
         network 10.0.0.0 0.0.0.255 area 0
        """
        result = count_tokens_tiktoken(config)
        # A real config should tokenize into multiple tokens
        assert result > 10, f"Expected >10 tokens, got {result}"

    def test_count_tokens_unknown_model_fallback(self):
        """Unknown model name should fall back to cl100k_base, not crash."""
        from token_calculator import count_tokens_tiktoken
        result = count_tokens_tiktoken("test input", model="nonexistent-model-xyz")
        assert isinstance(result, int)
        assert result > 0

    def test_get_token_breakdown(self):
        """Verify token breakdown returns (text, id) tuples."""
        from token_calculator import get_token_breakdown
        tokens = get_token_breakdown("show ip route")
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        # Each entry should be a (text, id) tuple
        for token_text, token_id in tokens:
            assert isinstance(token_text, str)
            assert isinstance(token_id, int)

    def test_get_token_breakdown_empty(self):
        """Edge case: empty string returns empty list."""
        from token_calculator import get_token_breakdown
        tokens = get_token_breakdown("")
        assert tokens == []


# ============================================================================
# DOCUMENT LOADER TESTS (Volume 2, Chapter 14)
# ============================================================================

class TestDocumentLoader:
    """
    Tests for document_loader.py — loads network docs for RAG.

    Like loading running-configs from a TFTP server, but for
    text/markdown/config files that feed into an AI knowledge base.
    """

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files, clean up after."""
        d = tempfile.mkdtemp()
        yield d
        shutil.rmtree(d)

    def test_load_text_file(self, temp_dir):
        """Load a plain text file and verify content + metadata."""
        from document_loader import load_text_file
        path = os.path.join(temp_dir, "test.txt")
        with open(path, "w") as f:
            f.write("show ip route\nshow ip bgp summary")

        result = load_text_file(path)
        assert "show ip route" in result["content"]
        assert result["metadata"]["type"] == "text"
        assert result["metadata"]["filename"] == "test.txt"

    def test_load_markdown_file_with_title(self, temp_dir):
        """Load markdown and extract h1 title."""
        from document_loader import load_markdown_file
        path = os.path.join(temp_dir, "guide.md")
        with open(path, "w") as f:
            f.write("# VLAN Configuration Guide\n\nVLAN 10 is for users.")

        result = load_markdown_file(path)
        assert result["metadata"]["title"] == "VLAN Configuration Guide"
        assert result["metadata"]["type"] == "markdown"

    def test_load_markdown_file_no_title(self, temp_dir):
        """Markdown without h1 should have title=None."""
        from document_loader import load_markdown_file
        path = os.path.join(temp_dir, "notes.md")
        with open(path, "w") as f:
            f.write("Just some notes without a heading.")

        result = load_markdown_file(path)
        assert result["metadata"]["title"] is None

    def test_load_config_file_with_hostname(self, temp_dir):
        """Load Cisco config and extract hostname."""
        from document_loader import load_config_file
        path = os.path.join(temp_dir, "router.cfg")
        with open(path, "w") as f:
            f.write("hostname CORE-RTR-01\ninterface Gi0/0\n ip address 10.0.0.1 255.255.255.0")

        result = load_config_file(path)
        assert result["metadata"]["hostname"] == "CORE-RTR-01"
        assert result["metadata"]["type"] == "config"

    def test_load_config_file_no_hostname(self, temp_dir):
        """Config without hostname line should have hostname=None."""
        from document_loader import load_config_file
        path = os.path.join(temp_dir, "partial.cfg")
        with open(path, "w") as f:
            f.write("interface Gi0/0\n ip address 10.0.0.1 255.255.255.0")

        result = load_config_file(path)
        assert result["metadata"]["hostname"] is None

    def test_load_docs_from_directory(self, temp_dir):
        """Load multiple files from a directory."""
        from document_loader import load_docs_from_directory

        # Create test files
        with open(os.path.join(temp_dir, "test.txt"), "w") as f:
            f.write("plain text content")
        with open(os.path.join(temp_dir, "guide.md"), "w") as f:
            f.write("# Guide\nSome content")
        with open(os.path.join(temp_dir, "router.cfg"), "w") as f:
            f.write("hostname RTR-01\ninterface Gi0/0")

        docs = load_docs_from_directory(temp_dir)
        assert len(docs) == 3
        types = {d["metadata"]["type"] for d in docs}
        assert types == {"text", "markdown", "config"}

    def test_load_docs_empty_directory(self, temp_dir):
        """Empty directory should return empty list, not crash."""
        from document_loader import load_docs_from_directory
        docs = load_docs_from_directory(temp_dir)
        assert docs == []

    def test_load_docs_nonexistent_directory(self):
        """Non-existent directory should return empty list."""
        from document_loader import load_docs_from_directory
        docs = load_docs_from_directory("/nonexistent/path")
        assert docs == []

    def test_create_sample_docs(self, temp_dir):
        """Verify sample doc creation creates expected files."""
        from document_loader import create_sample_docs
        output = os.path.join(temp_dir, "samples")
        create_sample_docs(output)

        files = os.listdir(output)
        assert "vlan_guide.md" in files
        assert "bgp_standards.md" in files
        assert "router_config.cfg" in files


# ============================================================================
# EVALUATION HARNESS TESTS (Volume 2, Chapter 26)
# ============================================================================

class TestEvalHarness:
    """
    Tests for eval_harness.py — validates AI-generated configs.

    Like a pre-deployment config audit: checks if the AI's output
    is safe, correct, and matches the expected configuration.
    """

    @pytest.fixture
    def evaluator(self):
        """Create a fresh evaluator instance."""
        from eval_harness import NetworkConfigEvaluator
        return NetworkConfigEvaluator()

    def test_exact_match_identical(self, evaluator):
        """Identical configs should match exactly."""
        assert evaluator._exact_match(
            "interface Gi0/0\n ip address 10.0.0.1 255.255.255.0",
            "interface Gi0/0\n ip address 10.0.0.1 255.255.255.0"
        )

    def test_exact_match_different(self, evaluator):
        """Different configs should NOT match."""
        assert not evaluator._exact_match(
            "interface Gi0/0",
            "interface Gi0/1"
        )

    def test_exact_match_whitespace_trim(self, evaluator):
        """Leading/trailing whitespace should be ignored."""
        assert evaluator._exact_match("  config  ", "config")

    def test_semantic_similarity_identical(self, evaluator):
        """Identical text → similarity = 1.0."""
        sim = evaluator._semantic_similarity(
            "router bgp 65001",
            "router bgp 65001"
        )
        assert sim == pytest.approx(1.0)

    def test_semantic_similarity_different(self, evaluator):
        """Completely different text → low similarity."""
        sim = evaluator._semantic_similarity(
            "interface GigabitEthernet0/0",
            "router ospf process-id 1"
        )
        assert sim < 0.5

    def test_semantic_similarity_empty(self, evaluator):
        """Empty strings → similarity = 0.0 (avoid division by zero)."""
        sim = evaluator._semantic_similarity("", "")
        assert sim == 0.0

    def test_validate_vendor_cisco(self, evaluator):
        """Cisco IOS config should pass Cisco syntax validation."""
        config = """
        interface GigabitEthernet0/0
         ip address 10.0.0.1 255.255.255.0
        router bgp 65001
        """
        assert evaluator._validate_vendor_syntax(config, "cisco")

    def test_validate_vendor_juniper(self, evaluator):
        """JunOS set commands should pass Juniper validation."""
        config = "set interfaces ge-0/0/0 unit 0 family inet address 10.0.0.1/24"
        assert evaluator._validate_vendor_syntax(config, "juniper")

    def test_validate_vendor_mismatch(self, evaluator):
        """Cisco config should fail Juniper validation."""
        config = "interface GigabitEthernet0/0\n ip address 10.0.0.1 255.255.255.0"
        # Juniper expects 'set interfaces', 'set protocols', etc.
        assert not evaluator._validate_vendor_syntax(config, "juniper")

    def test_is_valid_json_good(self, evaluator):
        """Valid JSON string should pass."""
        assert evaluator._is_valid_json('{"hostname": "RTR-01", "interfaces": 48}')

    def test_is_valid_json_bad(self, evaluator):
        """Invalid JSON should fail, not crash."""
        assert not evaluator._is_valid_json("not json at all")

    def test_safety_permit_any_any(self, evaluator):
        """ACL with 'permit any any' must fail safety check — this is dangerous."""
        config = "access-list 100 permit any any"
        result = evaluator.evaluate(
            model_output=config,
            ground_truth="access-list 100 deny ip any any",
            criteria=["No 'permit any any'", "Has default deny"],
            vendor="cisco"
        )
        # 'permit any any' is unsafe → should be flagged CRITICAL
        assert result.is_safe is False
        assert result.severity.value == "critical"

    def test_safety_telnet_enabled(self, evaluator):
        """Config with telnet enabled must fail 'no telnet' criterion."""
        config = "line vty 0 4\n transport input telnet"
        result = evaluator.evaluate(
            model_output=config,
            ground_truth="line vty 0 4\n transport input ssh",
            criteria=["No telnet enabled"],
            vendor="cisco"
        )
        # The 'no telnet' criterion should fail
        assert result.rubric_score == 0

    def test_evaluate_full_pass(self, evaluator):
        """Config meeting all criteria should pass."""
        config = """
        router bgp 65001
         neighbor 10.0.0.2 remote-as 65002
        access-list 100 deny ip any any
        """
        result = evaluator.evaluate(
            model_output=config,
            ground_truth=config,
            criteria=["BGP configured", "Has default deny"],
            vendor="cisco"
        )
        assert result.exact_match is True
        assert result.rubric_score == 2
        assert result.severity.value == "pass"

    def test_generate_verdict_unsafe(self, evaluator):
        """Unsafe config should get CRITICAL verdict."""
        verdict, severity = evaluator._generate_verdict(
            exact_match=True,
            semantic_sim=1.0,
            rubric_score=5,
            rubric_max=5,
            is_safe=False
        )
        from eval_harness import Severity
        assert severity == Severity.CRITICAL
        assert "CRITICAL" in verdict

    def test_generate_verdict_perfect(self, evaluator):
        """Perfect match should get PASS verdict."""
        verdict, severity = evaluator._generate_verdict(
            exact_match=True,
            semantic_sim=1.0,
            rubric_score=5,
            rubric_max=5,
            is_safe=True
        )
        from eval_harness import Severity
        assert severity == Severity.PASS
        assert "PERFECT" in verdict


# ============================================================================
# LOG TRIAGE TESTS (Volume 2, Chapter 27)
# ============================================================================

class TestLogTriage:
    """
    Tests for log_triage.py — classifies and correlates syslog messages.

    Think of this as an AI-powered syslog server: it reads raw messages
    like '%LINK-3-UPDOWN' and groups them into actionable incidents.
    """

    @pytest.fixture
    def classifier(self):
        """Create a fresh LogTriageClassifier."""
        from log_triage import LogTriageClassifier
        return LogTriageClassifier()

    def _make_log(self, message, hostname="CORE-RTR-01", timestamp="2026-01-15T10:00:00"):
        """Helper: create a raw log dict (like a syslog entry)."""
        return {
            "timestamp": timestamp,
            "hostname": hostname,
            "message": message,
            "raw": f"Jan 15 10:00:00 {hostname}: {message}"
        }

    def test_classify_auth_failure(self, classifier):
        """Authentication failure should be classified correctly."""
        logs = [self._make_log("SSH authentication failed for user admin from 192.168.1.100")]
        results = classifier.classify_logs(logs)
        assert len(results) == 1
        assert results[0].category == "authentication_failure"
        assert results[0].method == "rule"
        assert results[0].confidence == 0.95

    def test_classify_bgp_down(self, classifier):
        """BGP neighbor down should be a routing_protocol_event."""
        logs = [self._make_log("BGP neighbor 10.0.0.2 down - hold time expired")]
        results = classifier.classify_logs(logs)
        assert results[0].category == "routing_protocol_event"

    def test_classify_interface_down(self, classifier):
        """Interface flap should be an interface_event."""
        logs = [self._make_log("GigabitEthernet0/0: interface down, link loss")]
        results = classifier.classify_logs(logs)
        assert results[0].category == "interface_event"

    def test_classify_cpu_high(self, classifier):
        """High CPU should be resource_exhaustion."""
        logs = [self._make_log("CPU utilization exceeded threshold: 95%")]
        results = classifier.classify_logs(logs)
        assert results[0].category == "resource_exhaustion"

    def test_classify_hardware_failure(self, classifier):
        """Fan failure should be hardware_failure."""
        logs = [self._make_log("Fan 1 failure detected - system restart imminent")]
        results = classifier.classify_logs(logs)
        assert results[0].category == "hardware_failure"

    def test_classify_security_event(self, classifier):
        """ACL denied should be security_event."""
        logs = [self._make_log("ACL denied 192.168.1.50 -> 10.0.0.1 port 22")]
        results = classifier.classify_logs(logs)
        assert results[0].category == "security_event"

    def test_classify_unknown_fallback(self, classifier):
        """Unknown message should fall back to AI classifier."""
        logs = [self._make_log("Some completely unrelated log message about weather")]
        results = classifier.classify_logs(logs)
        assert results[0].method == "embedding"
        assert results[0].confidence == 0.75

    def test_severity_critical_on_restart(self, classifier):
        """Message containing 'restart' should be CRITICAL."""
        logs = [self._make_log("System restart in progress")]
        results = classifier.classify_logs(logs)
        assert results[0].severity == "critical"

    def test_severity_bgp_is_error(self, classifier):
        """BGP event should be ERROR severity by default."""
        logs = [self._make_log("BGP neighbor flap detected")]
        results = classifier.classify_logs(logs)
        assert results[0].severity == "error"

    def test_correlate_related_events(self, classifier):
        """Events within time window from same host should correlate."""
        from log_triage import LogEvent
        events = [
            LogEvent(
                timestamp="2026-01-15T10:00:00",
                hostname="CORE-RTR-01",
                message="BGP neighbor down",
                raw="test",
                category="routing_protocol_event",
                severity="error",
                confidence=0.95
            ),
            LogEvent(
                timestamp="2026-01-15T10:01:00",
                hostname="CORE-RTR-01",
                message="Interface GigabitEthernet0/0 down",
                raw="test",
                category="interface_event",
                severity="error",
                confidence=0.95
            ),
        ]
        groups = classifier.correlate_events(events, time_window_secs=300)
        assert len(groups) == 1
        assert len(groups[0].related_events) == 2

    def test_correlate_distant_events_separate(self, classifier):
        """Events outside time window should NOT correlate."""
        from log_triage import LogEvent
        events = [
            LogEvent(
                timestamp="2026-01-15T10:00:00",
                hostname="RTR-01",
                message="BGP neighbor down",
                raw="test",
                category="routing_protocol_event",
                severity="error",
                confidence=0.95
            ),
            LogEvent(
                timestamp="2026-01-15T12:00:00",  # 2 hours later
                hostname="RTR-02",
                message="Fan failure",
                raw="test",
                category="hardware_failure",
                severity="critical",
                confidence=0.95
            ),
        ]
        groups = classifier.correlate_events(events, time_window_secs=300)
        # Should create separate groups (2 hours apart)
        assert len(groups) == 2

    def test_temporal_relation_within_window(self, classifier):
        """Two events 60 seconds apart should be temporally related."""
        from log_triage import LogEvent
        log1 = LogEvent(
            timestamp="2026-01-15T10:00:00", hostname="R1",
            message="test", raw="test"
        )
        log2 = LogEvent(
            timestamp="2026-01-15T10:01:00", hostname="R1",
            message="test", raw="test"
        )
        assert classifier._are_temporally_related(log1, log2, 300)

    def test_temporal_relation_outside_window(self, classifier):
        """Two events 10 minutes apart with 5min window should NOT relate."""
        from log_triage import LogEvent
        log1 = LogEvent(
            timestamp="2026-01-15T10:00:00", hostname="R1",
            message="test", raw="test"
        )
        log2 = LogEvent(
            timestamp="2026-01-15T10:10:00", hostname="R1",
            message="test", raw="test"
        )
        assert not classifier._are_temporally_related(log1, log2, 300)

    def test_temporal_relation_bad_timestamp(self, classifier):
        """Invalid timestamp should return False, not crash."""
        from log_triage import LogEvent
        log1 = LogEvent(
            timestamp="not-a-date", hostname="R1",
            message="test", raw="test"
        )
        log2 = LogEvent(
            timestamp="also-not-a-date", hostname="R1",
            message="test", raw="test"
        )
        assert not classifier._are_temporally_related(log1, log2, 300)

    def test_classify_multiple_logs(self, classifier):
        """Batch classification of mixed log types."""
        logs = [
            self._make_log("SSH authentication failed for admin", timestamp="2026-01-15T10:00:00"),
            self._make_log("BGP neighbor 10.0.0.2 down", timestamp="2026-01-15T10:00:30"),
            self._make_log("CPU utilization high at 98%", timestamp="2026-01-15T10:01:00"),
        ]
        results = classifier.classify_logs(logs)
        assert len(results) == 3
        categories = [r.category for r in results]
        assert "authentication_failure" in categories
        assert "routing_protocol_event" in categories
        assert "resource_exhaustion" in categories

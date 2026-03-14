"""
Unit Tests for Chapter 6 — Structured Outputs
===============================================
Tests pure functions from Vol1_Ch6_Structured_Outputs.ipynb
that work WITHOUT API keys.

Covers:
  - parse_findings_v1:     Fragile text parser (Demo 1)
  - extract_json_from_text: Robust JSON extractor (Demo 2, V3)
  - validate_audit_logic:   Business logic validator (Demo 7)
  - Pydantic models:        NetworkDevice, Interface, AuditReport (Demos 4/8)

Run with:
    pytest tests/test_ch6_structured_outputs.py -v

Author: AI NetOps QA Agent
"""

import json
import re
import ipaddress
import pytest
from pydantic import BaseModel, ValidationError, field_validator
from typing import Optional, List
from enum import Enum


# ============================================================================
# FUNCTION DEFINITIONS (extracted from notebook — no API key needed)
# ============================================================================

def parse_findings_v1(llm_response: str) -> list:
    """V1: Only handles dash-prefixed lines. Silently drops other formats."""
    findings = []
    for line in llm_response.split('\n'):
        if line.strip().startswith('- '):
            findings.append(line.strip()[2:])
    return findings


def extract_json_from_text(text: str):
    """Extract JSON from LLM response — handles raw, markdown, and prose."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for pattern in [r'```json\s*([\s\S]*?)\s*```', r'```\s*([\s\S]*?)\s*```']:
        match = re.search(pattern, text)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                continue
    for open_c, close_c in [('[', ']'), ('{', '}')]:
        start = text.find(open_c)
        if start == -1:
            continue
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == open_c:
                depth += 1
            elif ch == close_c:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i+1])
                    except json.JSONDecodeError:
                        break
    return None


def validate_audit_logic(findings: dict) -> None:
    """Business logic: risk score vs. critical findings consistency."""
    critical_count = len(findings.get('critical_findings', []))
    risk_score = findings.get('overall_risk_score', 0)
    if critical_count > 0 and risk_score < 30:
        raise ValueError(
            f"Logic error: {critical_count} critical findings but risk score = {risk_score}. "
            "Risk score must be >= 30 when critical issues exist."
        )
    for severity in ['critical_findings', 'high_findings']:
        for f in findings.get(severity, []):
            if not f.get('remediation', '').strip():
                raise ValueError(f"Finding '{f.get('title','?')}' has empty remediation.")


# ── Pydantic models (from Demos 4 and 8) ─────────────────────────────────────

class NetworkDevice(BaseModel):
    hostname: str
    ip_address: str
    vendor: str


class InterfaceStatus(str, Enum):
    UP = "up"
    DOWN = "down"
    ADMIN_DOWN = "admin-down"


class Interface(BaseModel):
    name: str
    ip_address: Optional[str] = None
    subnet_mask: Optional[str] = None
    description: Optional[str] = None
    status: InterfaceStatus

    @field_validator('ip_address')
    @classmethod
    def validate_ip(cls, v):
        if v is None:
            return v
        try:
            ipaddress.IPv4Address(v)
            return v
        except ValueError:
            raise ValueError(f"Invalid IP address format: '{v}'")

    def cidr(self) -> Optional[str]:
        if not self.ip_address or not self.subnet_mask:
            return None
        try:
            net = ipaddress.IPv4Network(f"0.0.0.0/{self.subnet_mask}", strict=False)
            return f"{self.ip_address}/{net.prefixlen}"
        except ValueError:
            return None


class SecurityFinding(BaseModel):
    title: str
    description: str
    remediation: str
    config_line: Optional[str] = None


class HighFinding(BaseModel):
    title: str
    remediation: str
    description: Optional[str] = None


class AuditReport(BaseModel):
    device_hostname: str
    platform: str
    critical_findings: List[SecurityFinding]
    high_findings: List[HighFinding]
    passed_checks: List[str]
    overall_risk_score: int
    audit_summary: str

    @property
    def risk_level(self) -> str:
        if self.overall_risk_score >= 70:
            return "CRITICAL"
        elif self.overall_risk_score >= 40:
            return "HIGH"
        elif self.overall_risk_score >= 20:
            return "MEDIUM"
        return "LOW"


# ============================================================================
# TESTS: parse_findings_v1 (Demo 1)
# ============================================================================

class TestParseFindingsV1:
    """Tests for the fragile V1 parser — demonstrates why structured outputs exist."""

    def test_dash_format_parsed(self):
        """V1 handles dash-prefixed lines correctly."""
        text = "- Telnet enabled\n- SNMP public\n- No encryption"
        result = parse_findings_v1(text)
        assert len(result) == 3
        assert result[0] == "Telnet enabled"
        assert result[2] == "No encryption"

    def test_numbered_format_missed(self):
        """V1 silently drops numbered-list format."""
        text = "1. Telnet enabled\n2. SNMP public\n3. No encryption"
        assert parse_findings_v1(text) == []

    def test_bullet_format_missed(self):
        """V1 silently drops bullet-point format."""
        text = "• Telnet\n• SNMP\n• No encryption"
        assert parse_findings_v1(text) == []

    def test_prose_format_missed(self):
        """V1 silently drops prose format."""
        text = "I found that Telnet is enabled and SNMP is insecure."
        assert parse_findings_v1(text) == []

    def test_empty_input(self):
        """V1 returns empty list on empty input."""
        assert parse_findings_v1("") == []

    def test_whitespace_only(self):
        """V1 returns empty list on whitespace-only input."""
        assert parse_findings_v1("   \n\n  ") == []

    def test_mixed_formats(self):
        """V1 only captures dash-prefixed lines from mixed content."""
        text = "Issues:\n1. First\n- Second\n• Third\n- Fourth"
        result = parse_findings_v1(text)
        assert result == ["Second", "Fourth"]

    def test_indented_dashes(self):
        """V1 handles indented dash lines (stripped whitespace)."""
        text = "  - Indented finding\n    - Deep finding"
        result = parse_findings_v1(text)
        assert len(result) == 2


# ============================================================================
# TESTS: extract_json_from_text (Demo 2, V3)
# ============================================================================

class TestExtractJsonFromText:
    """Tests for the robust JSON extractor — the multi-protocol parser."""

    def test_raw_json_array(self):
        """Path 1: parse raw JSON array directly."""
        text = '[{"name": "Gi0/0", "status": "up"}]'
        result = extract_json_from_text(text)
        assert result == [{"name": "Gi0/0", "status": "up"}]

    def test_raw_json_object(self):
        """Path 1: parse raw JSON object directly."""
        text = '{"hostname": "router-01"}'
        result = extract_json_from_text(text)
        assert result == {"hostname": "router-01"}

    def test_markdown_json_block(self):
        """Path 2: extract JSON from ```json ... ``` block."""
        text = '```json\n[{"a": 1}]\n```'
        result = extract_json_from_text(text)
        assert result == [{"a": 1}]

    def test_markdown_plain_block(self):
        """Path 2: extract JSON from ``` ... ``` block (no json tag)."""
        text = '```\n{"b": 2}\n```'
        result = extract_json_from_text(text)
        assert result == {"b": 2}

    def test_json_in_prose(self):
        """Path 3: extract JSON embedded in explanation text."""
        text = 'Here are the interfaces: [{"name": "Gi0/0"}]'
        result = extract_json_from_text(text)
        assert result == [{"name": "Gi0/0"}]

    def test_json_object_in_prose(self):
        """Path 3: extract JSON object embedded in explanation text."""
        text = 'The result is: {"hostname": "sw-01"} as shown.'
        result = extract_json_from_text(text)
        assert result == {"hostname": "sw-01"}

    def test_no_json(self):
        """All paths fail — returns None, not crash."""
        assert extract_json_from_text("no json here") is None

    def test_empty_string(self):
        """Empty input returns None."""
        assert extract_json_from_text("") is None

    def test_whitespace_only(self):
        """Whitespace-only returns None."""
        assert extract_json_from_text("   \n\t  ") is None

    def test_whitespace_around_json(self):
        """JSON with surrounding whitespace is parsed correctly."""
        text = '  \n  {"key": "value"}  \n  '
        result = extract_json_from_text(text)
        assert result == {"key": "value"}

    def test_nested_json(self):
        """Handles nested JSON structures."""
        text = '{"a": {"b": [1, 2, 3]}}'
        result = extract_json_from_text(text)
        assert result == {"a": {"b": [1, 2, 3]}}

    def test_markdown_with_explanation(self):
        """Markdown block with surrounding explanation text."""
        text = 'Here is the result:\n```json\n{"x": 1}\n```\nDone.'
        result = extract_json_from_text(text)
        assert result == {"x": 1}


# ============================================================================
# TESTS: validate_audit_logic (Demo 7)
# ============================================================================

class TestValidateAuditLogic:
    """Tests for business logic validation — the domain-level sanity check."""

    def test_valid_high_risk(self):
        """Critical findings + high risk score = valid."""
        findings = {
            "critical_findings": [{"title": "Telnet", "remediation": "Use SSH"}],
            "overall_risk_score": 75
        }
        validate_audit_logic(findings)  # Should not raise

    def test_valid_no_findings(self):
        """No findings + low score = valid."""
        validate_audit_logic({"overall_risk_score": 0})

    def test_valid_empty_dict(self):
        """Empty dict = valid (defaults to 0 findings, 0 score)."""
        validate_audit_logic({})

    def test_invalid_low_score_with_critical(self):
        """Critical findings + low risk score = logic error."""
        findings = {
            "critical_findings": [{"title": "Telnet", "remediation": "Use SSH"}],
            "overall_risk_score": 5
        }
        with pytest.raises(ValueError, match="Logic error"):
            validate_audit_logic(findings)

    def test_invalid_empty_remediation(self):
        """Finding with empty remediation = rejected."""
        findings = {
            "critical_findings": [{"title": "Telnet", "remediation": ""}],
            "overall_risk_score": 80
        }
        with pytest.raises(ValueError, match="empty remediation"):
            validate_audit_logic(findings)

    def test_invalid_whitespace_remediation(self):
        """Finding with whitespace-only remediation = rejected."""
        findings = {
            "critical_findings": [{"title": "Telnet", "remediation": "   "}],
            "overall_risk_score": 80
        }
        with pytest.raises(ValueError, match="empty remediation"):
            validate_audit_logic(findings)

    def test_high_findings_checked(self):
        """High findings also checked for empty remediation."""
        findings = {
            "high_findings": [{"title": "Weak password", "remediation": ""}],
            "overall_risk_score": 50
        }
        with pytest.raises(ValueError, match="empty remediation"):
            validate_audit_logic(findings)

    def test_boundary_score_30(self):
        """Risk score exactly 30 with critical findings = valid."""
        findings = {
            "critical_findings": [{"title": "Telnet", "remediation": "Use SSH"}],
            "overall_risk_score": 30
        }
        validate_audit_logic(findings)  # Should not raise

    def test_boundary_score_29(self):
        """Risk score 29 with critical findings = logic error."""
        findings = {
            "critical_findings": [{"title": "Telnet", "remediation": "Use SSH"}],
            "overall_risk_score": 29
        }
        with pytest.raises(ValueError, match="Logic error"):
            validate_audit_logic(findings)


# ============================================================================
# TESTS: Pydantic Models (Demos 4 and 8)
# ============================================================================

class TestNetworkDevice:
    """Tests for the simplest Pydantic model — three required fields."""

    def test_valid_device(self):
        dev = NetworkDevice(hostname="sw-01", ip_address="10.0.0.1", vendor="Cisco")
        assert dev.hostname == "sw-01"
        assert dev.ip_address == "10.0.0.1"

    def test_missing_field(self):
        with pytest.raises(ValidationError):
            NetworkDevice(hostname="sw-01", vendor="Cisco")

    def test_from_dict(self):
        raw = {"hostname": "sw-01", "ip_address": "10.0.0.1", "vendor": "Cisco"}
        dev = NetworkDevice(**raw)
        assert dev.hostname == "sw-01"


class TestInterface:
    """Tests for the interface model with Enum status and IP validation."""

    def test_valid_interface(self):
        iface = Interface(
            name="Gi0/0", ip_address="10.0.0.1",
            subnet_mask="255.255.255.0", status="up"
        )
        assert iface.status == InterfaceStatus.UP
        assert iface.cidr() == "10.0.0.1/24"

    def test_optional_fields(self):
        iface = Interface(name="Gi0/1", status="down")
        assert iface.ip_address is None
        assert iface.cidr() is None

    def test_invalid_ip(self):
        with pytest.raises(ValidationError, match="Invalid IP"):
            Interface(name="Gi0/0", ip_address="999.999.999.999", status="up")

    def test_invalid_status(self):
        with pytest.raises(ValidationError):
            Interface(name="Gi0/0", status="active")

    def test_admin_down(self):
        iface = Interface(name="Gi0/2", status="admin-down")
        assert iface.status == InterfaceStatus.ADMIN_DOWN

    def test_cidr_slash30(self):
        iface = Interface(
            name="Gi0/0", ip_address="203.0.113.1",
            subnet_mask="255.255.255.252", status="up"
        )
        assert iface.cidr() == "203.0.113.1/30"

    def test_cidr_loopback(self):
        iface = Interface(
            name="Lo0", ip_address="172.16.255.1",
            subnet_mask="255.255.255.255", status="up"
        )
        assert iface.cidr() == "172.16.255.1/32"


class TestAuditReport:
    """Tests for the full audit report model — the production data contract."""

    @pytest.fixture
    def sample_report_data(self):
        return {
            "device_hostname": "branch-fw-01",
            "platform": "IOS",
            "critical_findings": [
                {"title": "Telnet", "description": "Telnet enabled", "remediation": "Use SSH"}
            ],
            "high_findings": [
                {"title": "Weak password", "remediation": "Use enable secret"}
            ],
            "passed_checks": ["SSH enabled"],
            "overall_risk_score": 75,
            "audit_summary": "High risk device"
        }

    def test_valid_report(self, sample_report_data):
        report = AuditReport(**sample_report_data)
        assert report.device_hostname == "branch-fw-01"
        assert report.critical_findings[0].title == "Telnet"
        assert report.high_findings[0].title == "Weak password"

    def test_risk_level_critical(self, sample_report_data):
        sample_report_data["overall_risk_score"] = 95
        assert AuditReport(**sample_report_data).risk_level == "CRITICAL"

    def test_risk_level_high(self, sample_report_data):
        sample_report_data["overall_risk_score"] = 50
        assert AuditReport(**sample_report_data).risk_level == "HIGH"

    def test_risk_level_medium(self, sample_report_data):
        sample_report_data["overall_risk_score"] = 25
        assert AuditReport(**sample_report_data).risk_level == "MEDIUM"

    def test_risk_level_low(self, sample_report_data):
        sample_report_data["overall_risk_score"] = 10
        assert AuditReport(**sample_report_data).risk_level == "LOW"

    def test_missing_required_field(self, sample_report_data):
        del sample_report_data["device_hostname"]
        with pytest.raises(ValidationError):
            AuditReport(**sample_report_data)

    def test_optional_config_line(self, sample_report_data):
        """config_line is optional on SecurityFinding."""
        sample_report_data["critical_findings"][0]["config_line"] = "enable password cisco"
        report = AuditReport(**sample_report_data)
        assert report.critical_findings[0].config_line == "enable password cisco"

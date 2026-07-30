from cyberqa.approval import ApprovalPolicy
from cyberqa.models import Event, Role


def test_autonomous_and_destructive_actions_are_distinct():
    policy = ApprovalPolicy()
    assert not policy.requires_approval("restart_service")
    assert policy.requires_approval("reset_credential")


def test_event_has_stable_contract():
    event = Event(type="LDAP_FAILED", run_id="r1", emitted_by=Role.VALIDATION)
    assert event.type == "LDAP_FAILED"
    assert event.id

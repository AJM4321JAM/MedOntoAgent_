"""Offline tests: protected workspace read models never leak source internals."""
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException
import pytest
from app.api import routes_workspace


def test_missing_registry_is_not_misrepresented_as_verified_empty_corpus():
    with patch.object(routes_workspace, '_read_registry', return_value=(False, [])):
        result = routes_workspace.read_library()
    assert result.registry_state == 'missing'
    assert result.total_documents == 0


def test_registry_errors_return_sanitized_503():
    with patch.object(routes_workspace, '_read_registry', side_effect=routes_workspace.DocumentRegistryError('private location')):
        with pytest.raises(HTTPException) as caught:
            routes_workspace.read_library()
    assert caught.value.status_code == 503
    assert 'private' not in caught.value.detail


def test_registry_allowlist_excludes_paths_checksum_and_error_details():
    doc = SimpleNamespace(document_id='DOC-TEST', title='Test', publisher='Sample', clinical_domain='testing',
        document_type='GUIDELINE', source_authority='GUIDELINE', version='v1', approval_status='APPROVED',
        index_status='INDEXED', publication_date=None, effective_date=None, registered_at=datetime.now(timezone.utc),
        indexed_at=None, source_url=None, supersedes_document_id=None, superseded_by_document_id=None,
        file_path='PRIVATE_PATH', checksum_sha256='SECRET_CHECKSUM', index_error='INTERNAL_ERROR')
    with patch.object(routes_workspace, '_read_registry', return_value=(True, [doc])):
        result = routes_workspace.read_library()
    body = result.model_dump_json()
    assert result.total_documents == 1
    for sensitive in ('PRIVATE_PATH', 'SECRET_CHECKSUM', 'INTERNAL_ERROR', 'file_path', 'checksum_sha256', 'index_error'):
        assert sensitive not in body


def test_operations_does_not_probe_dependencies_or_export_secrets():
    result = routes_workspace.read_operations()
    body = result.model_dump_json()
    assert result.status == 'running'
    assert result.historical_metrics_available is False
    assert all(component.state == 'not_checked' for component in result.components if component.label in ('Qdrant Cloud', 'Gemini'))
    assert not any(k in body.lower() for k in ('api_key', 'secret', 'password', 'token'))

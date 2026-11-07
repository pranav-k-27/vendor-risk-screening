import shutil
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    """Every test gets its own data dir; nothing touches the real ./data."""
    data = tmp_path / "data"
    (data / "sanctions").mkdir(parents=True)
    for f in (FIXTURES / "sanctions").iterdir():
        shutil.copy(f, data / "sanctions" / f.name)
    (data / "documents").mkdir()
    monkeypatch.setenv("VRS_DATA_DIR", str(data))
    monkeypatch.setenv("VRS_SEARCH_MODE", "replay")
    yield data


@pytest.fixture
def snapshots(isolated_data_dir):
    from core.snapshot_manager import SnapshotManager

    return SnapshotManager()


@pytest.fixture
def entity():
    from core.schemas import EntityResolution, EntityStatus

    return EntityResolution(
        status=EntityStatus.RESOLVED,
        input_name="Acme Infotech Ltd",
        resolved_name="ACME INFOTECH LIMITED",
        lei="984500ABCDEF12345678",
        aliases=["Acme Infotech"],
        confidence=0.97,
        method="test",
    )

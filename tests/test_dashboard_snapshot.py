import shutil

import duckdb
from dashboard_fixture import build
from streamlit.testing.v1 import AppTest
from test_dashboard_app import APP

from cryptoval.dashboard import snapshot


def test_hosted_app_renders_from_the_published_snapshot(tmp_path, monkeypatch):
    db = str(tmp_path / "full.duckdb")
    con = duckdb.connect(db)
    build(con)
    con.close()
    # the publish tree: the app beside the Parquet the daily run exports, and no data root
    (tmp_path / "dashboard").mkdir()
    shutil.copy(APP, tmp_path / "dashboard" / "app.py")
    snapshot.export(db, str(tmp_path / "snapshot"))
    monkeypatch.delenv("CRYPTOVAL_DATA_ROOT", raising=False)
    at = AppTest.from_file(str(tmp_path / "dashboard" / "app.py"), default_timeout=60).run()
    assert not at.exception, at.exception
    assert not at.error
    assert list(at.dataframe[0].value["symbol"]) == ["USDC", "BTC"]
    snapshot.build(str(tmp_path / "snapshot"), f"{tmp_path}/rebuilt.duckdb")
    con = duckdb.connect(db, read_only=True)
    con.execute(f"attach '{tmp_path}/rebuilt.duckdb' as rebuilt (read_only)")
    for t in snapshot.TABLES:
        n = con.execute(f"select count(*) from {t}").fetchone()[0]
        diff = f"(select * from {t} except all select * from rebuilt.{t})"
        assert n > 0 and con.execute(f"select count(*) from {diff}").fetchone()[0] == 0, t

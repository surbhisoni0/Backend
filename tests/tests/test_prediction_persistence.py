"""One persisted prediction per target; a finished backtest must not replace it; a new result opens a new target."""
from pathlib import Path
from app.database.factory import create_repository
from app.prediction.engine import PredictionEngine
from app.services.predictor_service import PredictorService
from app.timing import ClockConfig

CSV = Path(__file__).resolve().parents[1] / "data" / "wingo_transcribed.csv"


def make():
    repo = create_repository("memory://"); repo.init()
    s = PredictorService(repo, PredictionEngine(), 100, 300, clock_cfg=ClockConfig())
    s.import_csv(CSV.read_text(encoding="utf-8"))
    return s, repo


def test_repeated_requests_return_same_saved_prediction():
    s, _ = make()
    a, b = s.next_prediction(), s.next_prediction()
    assert (a["id"], a["created_at"], a["target_seq"]) == (b["id"], b["created_at"], b["target_seq"])


def test_backtest_does_not_regenerate_pending_prediction():
    s, repo = make()
    before = s.next_prediction()
    s.run_backtest(use_ml=False)
    after = s.next_prediction()
    assert (before["id"], before["created_at"]) == (after["id"], after["created_at"])
    rows, total = repo.list_predictions(50, 0)
    assert total == 1 and [r["target_seq"] for r in rows] == [before["target_seq"]]


def test_new_result_resolves_old_target_and_opens_new_one_without_duplicates():
    s, repo = make()
    first = s.next_prediction()
    out = s.add_result(3)
    assert out["resolved_prediction"]["target_seq"] == first["target_seq"]
    nxt = out["next_prediction"]
    assert nxt["target_seq"] == first["target_seq"] + 1 and nxt["id"] != first["id"]
    assert s.next_prediction()["id"] == nxt["id"]
    _, total = repo.list_predictions(50, 0)
    assert total == 2

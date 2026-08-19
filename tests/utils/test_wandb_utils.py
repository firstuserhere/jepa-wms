import re

import pytest

from src.utils.wandb_utils import generate_wandb_run_id


def test_generate_wandb_run_id_is_safe_and_unique():
    run_ids = {generate_wandb_run_id() for _ in range(100)}

    assert len(run_ids) == 100
    assert all(re.fullmatch(r"[a-z0-9]{24}", run_id) for run_id in run_ids)


def test_generate_wandb_run_id_rejects_too_short_ids():
    with pytest.raises(ValueError, match="at least 8"):
        generate_wandb_run_id(7)

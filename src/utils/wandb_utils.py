"""Small W&B compatibility helpers with no dependency on private SDK APIs."""

import secrets
import string


_RUN_ID_ALPHABET = string.ascii_lowercase + string.digits


def generate_wandb_run_id(length: int = 24) -> str:
    """Return a W&B-safe, cryptographically random run identifier."""

    if length < 8:
        raise ValueError("W&B run IDs must contain at least 8 characters")
    return "".join(secrets.choice(_RUN_ID_ALPHABET) for _ in range(length))

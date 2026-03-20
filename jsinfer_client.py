"""Shared jsinfer client wrapper with slower polling and light rate-limit handling."""

from __future__ import annotations

import asyncio
import os
import time

from jsinfer import BatchInferenceClient

KEY_1 = "4adeb4ee-43c0-43a5-bbf2-b56977001584"
KEY_2 = "1c1abda6-2afc-49b1-b431-c6a5b0e412ec"

DEFAULT_API_KEY = (
    os.environ.get("JSINFER_API_KEY")
    or os.environ.get("DORMANT_API_KEY")
    or KEY_2
)

FAILED_BATCH_STATUSES = {"failed", "cancelled", "expired", "error"}


class JSInferClient(BatchInferenceClient):
    """BatchInferenceClient with repo-standard polling defaults."""

    def __init__(
        self,
        api_key: str | None = None,
        poll_interval_s: float = 10.0,
        rate_limit_backoff_s: float = 30.0,
    ):
        super().__init__(api_key=api_key or DEFAULT_API_KEY)
        self.poll_interval_s = poll_interval_s
        self.rate_limit_backoff_s = rate_limit_backoff_s

    async def poll_batch(self, batch_id: str, timeout: int = 60 * 60 * 24):
        start_time = time.time()
        last_status = None

        while time.time() - start_time < timeout:
            try:
                batch = await self.get_batch(batch_id)
            except Exception as exc:  # pragma: no cover - depends on remote API
                if "429" in str(exc):
                    print(
                        f"[jsinfer] batch={batch_id} rate limited; "
                        f"sleeping {self.rate_limit_backoff_s:.0f}s"
                    )
                    await asyncio.sleep(self.rate_limit_backoff_s)
                    continue
                raise

            batch_meta = batch.get("batch", {})
            status = batch_meta.get("status")

            if status == "completed":
                return batch["resultsUrl"]

            if status in FAILED_BATCH_STATUSES:
                errors = batch_meta.get("errors")
                raise RuntimeError(
                    f"Batch {batch_id} failed with status={status}. Errors: {errors}"
                )

            if status and status != last_status:
                print(f"[jsinfer] batch={batch_id} status={status}")
                last_status = status

            await asyncio.sleep(self.poll_interval_s)

        raise TimeoutError(f"Batch {batch_id} timed out after {timeout} seconds")


def create_client(
    api_key: str | None = None,
    poll_interval_s: float = 10.0,
    rate_limit_backoff_s: float = 30.0,
) -> JSInferClient:
    """Construct the repo-standard jsinfer client."""

    return JSInferClient(
        api_key=api_key,
        poll_interval_s=poll_interval_s,
        rate_limit_backoff_s=rate_limit_backoff_s,
    )

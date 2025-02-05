from typing import Optional

from autoqml.backend import Provider, Backend


def list_available_backends(
    provider: Optional[Provider] = None
) -> set[Backend]:
    # 1. Collect simulator backends
    # 2. If provider is given, collect available online backends
    # 3. Return joined set of backends
    pass

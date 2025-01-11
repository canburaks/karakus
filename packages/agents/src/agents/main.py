from dataclasses import dataclass

import httpx


@dataclass
class MyDeps:
    http_client: httpx.AsyncClient

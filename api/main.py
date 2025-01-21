from typing import Union

import uvicorn
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles

from .middlewares.api_error_middleware import APIErrorMiddleware
from .routers import auth_router, users_router
from .routers.ai import router as ai_router
from .core.method_test import method_test

app = FastAPI()

app.add_middleware(APIErrorMiddleware)

app.include_router(router=users_router)
app.include_router(router=auth_router)
app.include_router(router=ai_router)


@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
async def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.get(path="/test")
async def test_route(request: Request):
    return await method_test(request=request)


app.mount(path="/static", app=StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload_dirs=["api/"],
    )

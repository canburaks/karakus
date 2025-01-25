from typing import Union

import uvicorn
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from .core.method_test import method_test
from .middlewares.api_error_middleware import APIErrorMiddleware
from .routers import auth_router, users_router
from .routers.ai import router as ai_router
from .middlewares.monitoring import setup_monitoring

app = FastAPI(
    title="AI API",
    description="API for AI services including text generation and embeddings",
    version="1.0.0",
)

app.add_middleware(APIErrorMiddleware)

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up monitoring
setup_monitoring(app)

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
        reload_excludes=[".venv/*"],
        timeout_keep_alive=300,  # Keep-alive timeout in seconds
    )

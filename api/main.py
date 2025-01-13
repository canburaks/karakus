from typing import Union

import uvicorn
from fastapi import Depends, FastAPI
from fastapi.staticfiles import StaticFiles

from .middlewares.api_error_middleware import APIErrorMiddleware
from .middlewares.auth_middleware import verify_token_balance
from .routers import auth_router, users_router
from .routers.ai import router as ai_router
from .routers.lemonsqueezy import router as lemonsqueezy_router

app = FastAPI()

app.add_middleware(APIErrorMiddleware)

# Add authentication to all routes except auth and webhooks
app.include_router(router=auth_router)
app.include_router(router=lemonsqueezy_router)  # Webhooks don't need auth
app.include_router(router=users_router, dependencies=[Depends(verify_token_balance)])
app.include_router(router=ai_router, dependencies=[Depends(verify_token_balance)])


@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
async def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


app.mount(path="/static", app=StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

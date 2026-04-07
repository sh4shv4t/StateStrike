from __future__ import annotations

"""FastAPI honeypot target for stateful API fuzzing experiments."""

import logging
import re
import time
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from honeypot.database import get_db, init_db
from honeypot.middleware import TelemetryMiddleware, create_telemetry_router
from honeypot.models import Order, User

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger(__name__)

app = FastAPI(title="StateStrike Honeypot", version="1.0.0")
app.add_middleware(TelemetryMiddleware)
app.include_router(create_telemetry_router())


class UserCreate(BaseModel):
    """Payload for POST /users."""

    email: str = Field(min_length=1, max_length=256)


class OrderCreate(BaseModel):
    """Payload for POST /orders."""

    user_id: int
    item: str = Field(min_length=1, max_length=256)


@app.on_event("startup")
async def on_startup() -> None:
    """Initialize database tables at service startup."""

    init_db()


@app.get("/health")
def health_check() -> dict[str, object]:
    """Return liveness and timestamp information.

    Returns:
        Dictionary with service status and UNIX timestamp.
    """

    return {"status": "ok", "ts": int(time.time())}


@app.post("/users")
def create_user(payload: UserCreate, db: Session = Depends(get_db)) -> dict[str, object]:
    """Create a user with intentionally vulnerable regex validation.

    Args:
        payload: User creation body.
        db: SQLAlchemy session.

    Returns:
        Created user dictionary.

    Raises:
        HTTPException: If email validation fails.
    """

    pattern = r"^([a-zA-Z0-9]+\s?)*[a-zA-Z0-9]+$"

    # VULNERABILITY: ReDoS via catastrophic backtracking
    # Reference: Davis et al., "ReDoS in the Wild" (USENIX Security 2018)
    # This pattern exhibits O(2^n) backtracking on input "aaa...a!"
    # A production-hardened alternative would use: re2 or a finite automaton
    if not re.fullmatch(pattern, payload.email, flags=re.DOTALL):
        raise HTTPException(status_code=400, detail="Invalid email format")

    user = User(email=payload.email)
    db.add(user)
    db.commit()
    db.refresh(user)

    return {"id": user.id, "email": user.email, "created_at": user.created_at.isoformat()}


@app.get("/users/{user_id}")
def get_user(user_id: int, db: Session = Depends(get_db)) -> dict[str, object]:
    """Fetch user by identifier.

    Args:
        user_id: User identifier.
        db: SQLAlchemy session.

    Returns:
        User dictionary.

    Raises:
        HTTPException: If user does not exist.
    """

    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return {"id": user.id, "email": user.email, "created_at": user.created_at.isoformat()}


@app.post("/orders")
def create_order(payload: OrderCreate, db: Session = Depends(get_db)) -> dict[str, object]:
    """Create an order for an existing user.

    Args:
        payload: Order creation body.
        db: SQLAlchemy session.

    Returns:
        Created order dictionary.

    Raises:
        HTTPException: If user does not exist.
    """

    user = db.query(User).filter(User.id == payload.user_id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    order = Order(user_id=payload.user_id, item=payload.item)
    db.add(order)
    db.commit()
    db.refresh(order)

    return {
        "id": order.id,
        "user_id": order.user_id,
        "item": order.item,
        "created_at": order.created_at.isoformat(),
    }


@app.get("/orders")
def list_orders(
    user_id: int | None = Query(default=None),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    """List orders and expose intentional stateful degradation path.

    Args:
        user_id: Optional user filter and degradation trigger key.
        db: SQLAlchemy session.

    Returns:
        Order list payload with count metadata.
    """

    query = db.query(Order)
    if user_id is not None:
        query = query.filter(Order.user_id == user_id)

    orders = query.all()

    if user_id is not None and len(orders) > 20:
        # VULNERABILITY: Unindexed aggregate query degradation
        # Only reachable after stateful chain: 21x POST /orders -> GET /orders
        # An RL agent can discover this; a stateless fuzzer cannot.
        # Reference: RESTler (Atlidakis et al., ICSE 2019) pioneered stateful
        # REST fuzzing but used grammar-based, not RL-based exploration.
        all_orders = db.query(Order).all()
        expensive_aggregate: dict[int, int] = {}
        for left in all_orders:
            total = 0
            for right in all_orders:
                if left.user_id == right.user_id:
                    total += 1
            expensive_aggregate[left.user_id] = total
        LOGGER.info(
            "Triggered synthetic O(n^2) aggregate for user_id=%s with %s total rows",
            user_id,
            len(all_orders),
        )
        time.sleep(0.8)

    return {
        "count": len(orders),
        "orders": [
            {
                "id": order.id,
                "user_id": order.user_id,
                "item": order.item,
                "created_at": order.created_at.isoformat()
                if isinstance(order.created_at, datetime)
                else datetime.now(timezone.utc).isoformat(),
            }
            for order in orders
        ],
    }

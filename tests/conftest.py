import logging
import sys
import types
import uuid
from datetime import datetime, timedelta
import os
import pytest

# Provide a lightweight stub for kalshi_python if it's not installed so tests can import mm.py
if 'kalshi_python' not in sys.modules:
    kalshi_python = types.ModuleType('kalshi_python')
    class _DummyConfig:
        pass
    class _DummyClient:
        def __init__(self, *args, **kwargs):
            pass
    kalshi_python.Configuration = _DummyConfig
    kalshi_python.KalshiClient = _DummyClient
    sys.modules['kalshi_python'] = kalshi_python

    models_mod = types.ModuleType('kalshi_python.models')
    create_order_request_mod = types.ModuleType('kalshi_python.models.create_order_request')
    class CreateOrderRequest:
        pass
    create_order_request_mod.CreateOrderRequest = CreateOrderRequest
    sys.modules['kalshi_python.models'] = models_mod
    sys.modules['kalshi_python.models.create_order_request'] = create_order_request_mod

# Ensure project root is importable for `import mm`
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mm import AvellanedaMarketMaker, AbstractTradingAPI, MetricsTracker


class DummyAPI(AbstractTradingAPI):
    """A controllable test double for the trading API."""

    def __init__(self, *, market_ticker: str = 'TEST', initial_prices=None, initial_position: int = 0, initial_orders=None):
        self.market_ticker = market_ticker
        self._prices = initial_prices or {"yes": 0.50, "no": 0.50}
        self._position = initial_position
        self._orders = list(initial_orders or [])
        self.placed_orders = []
        self.canceled_orders = []

    # --- API surface used by the market maker ---
    def get_price(self):
        return dict(self._prices)

    def place_order(self, action: str, side: str, price: float, quantity: int, expiration_ts: int = None) -> str:
        order_id = str(uuid.uuid4())
        cents = int(round(price * 100))
        price_field = 'yes_price' if side == 'yes' else 'no_price'
        order = {
            'order_id': order_id,
            'ticker': self.market_ticker,
            'action': action,
            'side': side,
            'type': 'limit',
            price_field: cents,
            'count': quantity,
            'remaining_count': quantity,
            'status': 'resting',
            'expiration_time': int(expiration_ts or 0),
            'created_time': datetime.utcnow().isoformat(),
            'updated_time': datetime.utcnow().isoformat(),
        }
        self._orders.append(order)
        self.placed_orders.append(order)
        return order_id

    def cancel_order(self, order_id: str) -> bool:
        before = len(self._orders)
        self._orders = [o for o in self._orders if str(o.get('order_id')) != str(order_id)]
        cancelled = before != len(self._orders)
        if cancelled:
            self.canceled_orders.append(order_id)
        return cancelled

    def get_position(self) -> int:
        return int(self._position)

    def get_orders(self):
        return list(self._orders)

    # --- helpers for tests ---
    def set_price(self, yes: float = None, no: float = None):
        if yes is not None:
            self._prices['yes'] = yes
        if no is not None:
            self._prices['no'] = no

    def set_position(self, position: int):
        self._position = int(position)

    def set_orders(self, orders):
        self._orders = list(orders)


@pytest.fixture()
def logger():
    lg = logging.getLogger('TestLogger')
    lg.setLevel(logging.INFO)
    # ensure clean handlers to avoid duplicate logs across tests
    if not any(isinstance(h, logging.NullHandler) for h in lg.handlers):
        lg.addHandler(logging.NullHandler())
    return lg


def make_market_maker(*, logger, api: DummyAPI, gamma: float = 0.01, k: float = 1.5,
                      max_position: int = 10, order_expiration: int = 15,
                      min_spread: float = 0.01, position_limit_buffer: float = 0.2,
                      inventory_skew_factor: float = 0.01, trade_side: str = 'yes') -> AvellanedaMarketMaker:
    mm = AvellanedaMarketMaker(
        logger=logger,
        api=api,
        gamma=gamma,
        k=k,
        max_position=max_position,
        order_expiration=order_expiration,
        min_spread=min_spread,
        position_limit_buffer=position_limit_buffer,
        inventory_skew_factor=inventory_skew_factor,
        trade_side=trade_side,
        stop_event=None,
    )
    # Make metrics available for assertions without running the warmup/run loop
    mm.metrics = MetricsTracker(strategy_name='TestStrategy', market_ticker=api.market_ticker)
    # Give a finite horizon for formulas using (T - t)
    mm.T = 3600
    # Set a default sigma for deterministic calculations unless tests override
    mm.sigma = 0.02
    return mm


@pytest.fixture()
def dummy_api():
    return DummyAPI(market_ticker='TEST', initial_prices={"yes": 0.50, "no": 0.50}, initial_position=0, initial_orders=[])


@pytest.fixture()
def maker(logger, dummy_api):
    return make_market_maker(logger=logger, api=dummy_api)



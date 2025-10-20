import abc
import time
from typing import Dict, List, Tuple, Optional
import threading
import requests
import logging
import uuid
import math
import os
from kalshi_python import Configuration, KalshiClient
from kalshi_python.models.create_order_request import CreateOrderRequest
import json

def to_tick(p: float) -> float:
    # clamp to valid cents 0.01..0.99 and ROUND, not floor
    p = max(0.01, min(0.99, round(p, 2)))
    return p

def to_cents(p: float) -> int:
    return int(round(to_tick(p) * 100))

class MetricsTracker:
    def __init__(self, strategy_name: str, market_ticker: str):
        self.strategy_name = strategy_name
        self.market_ticker = market_ticker
        self.start_time = time.time()
        self.loop_snapshots: List[Dict] = []
        self.action_log: List[Dict] = []
        self.latencies: List[Dict] = []

    def record_loop(self, t_seconds: float, mid_price: float, inventory: int,
                    reservation_price: float, bid_price: float, ask_price: float,
                    buy_size: int, sell_size: int) -> None:
        self.loop_snapshots.append({
            "t_seconds": round(t_seconds, 3),
            "mid_price": round(mid_price, 4),
            "inventory": int(inventory),
            "reservation_price": round(reservation_price, 4),
            "bid_price": round(bid_price, 4),
            "ask_price": round(ask_price, 4),
            "buy_size": int(buy_size),
            "sell_size": int(sell_size),
        })

    def record_action(self, kind: str, details: Dict) -> None:
        entry = {"ts": time.time(), "kind": kind}
        entry.update(details or {})
        self.action_log.append(entry)

    def record_latency(self, name: str, seconds: float) -> None:
        self.latencies.append({
            "ts": time.time(),
            "name": name,
            "ms": round(seconds * 1000.0, 2)
        })

    def summarize(self) -> Dict:
        runtime_s = time.time() - self.start_time
        orders_placed = sum(1 for a in self.action_log if a.get("kind") == "place_order")
        orders_canceled = sum(1 for a in self.action_log if a.get("kind") == "cancel_order")
        orders_kept = sum(1 for a in self.action_log if a.get("kind") == "keep_order")
        orders_skipped = sum(1 for a in self.action_log if a.get("kind") == "skip_place")
        last_inventory = self.loop_snapshots[-1]["inventory"] if self.loop_snapshots else 0
        return {
            "strategy_name": self.strategy_name,
            "market_ticker": self.market_ticker,
            "runtime_seconds": round(runtime_s, 3),
            "num_iterations": len(self.loop_snapshots),
            "orders_placed": orders_placed,
            "orders_canceled": orders_canceled,
            "orders_kept": orders_kept,
            "orders_skipped": orders_skipped,
            "final_inventory": last_inventory,
        }

    def export_files(self, base_prefix: str) -> None:
        try:
            summary = self.summarize()
            payload = {
                "summary": summary,
                "loop_snapshots": self.loop_snapshots,
                "action_log": self.action_log,
                "latencies": self.latencies,
            }
            with open(f"{base_prefix}_metrics.json", "w") as f:
                json.dump(payload, f, indent=2)

            # Loops CSV
            try:
                with open(f"{base_prefix}_loops.csv", "w") as f:
                    f.write("t_seconds,mid_price,inventory,reservation_price,bid_price,ask_price,buy_size,sell_size\n")
                    for s in self.loop_snapshots:
                        f.write(
                            f"{s['t_seconds']},{s['mid_price']},{s['inventory']},{s['reservation_price']},{s['bid_price']},{s['ask_price']},{s['buy_size']},{s['sell_size']}\n"
                        )
            except Exception:
                pass

            # Actions CSV
            try:
                with open(f"{base_prefix}_actions.csv", "w") as f:
                    f.write("ts,kind,order_id,action,side,price,size,reason\n")
                    for a in self.action_log:
                        f.write(
                            f"{a.get('ts','')},{a.get('kind','')},{a.get('order_id','')},{a.get('action','')},{a.get('side','')},{a.get('price','')},{a.get('size','')},{a.get('reason','')}\n"
                        )
            except Exception:
                pass

            # Latencies CSV
            try:
                with open(f"{base_prefix}_latencies.csv", "w") as f:
                    f.write("ts,name,ms\n")
                    for l in self.latencies:
                        f.write(f"{l.get('ts','')},{l.get('name','')},{l.get('ms','')}\n")
            except Exception:
                pass
        except Exception:
            # Avoid raising during shutdown
            pass

class AbstractTradingAPI(abc.ABC):
    @abc.abstractmethod
    def get_price(self) -> float:
        pass

    @abc.abstractmethod
    def place_order(self, action: str, side: str, price: float, quantity: int, expiration_ts: int = None) -> str:
        pass

    @abc.abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        pass

    @abc.abstractmethod
    def get_position(self) -> int:
        pass

    @abc.abstractmethod
    def get_orders(self) -> List[Dict]:
        pass

class KalshiTradingAPI(AbstractTradingAPI):
    def __init__(
        self,
        email: str,
        password: str,
        market_ticker: str,
        base_url: str,
        logger: logging.Logger,
    ):
        self.email = email
        self.password = password
        self.market_ticker = market_ticker
        self.token = None
        self.member_id = None
        self.logger = logger
        self.base_url = base_url
        self.login()

    def login(self):
        self.logger.info("Logging in...")
        config = Configuration()
        with open(os.getenv("KALSHI_PRIVATE_KEY_PATH"), "r") as f:
            private_key = f.read()
        config.api_key_id = os.getenv("KALSHI_API_KEY_ID")
        config.private_key_pem = private_key
        self.client = KalshiClient(config)
        balance = self.client.get_balance()
        self.logger.info(f"Balance: {balance}")
        self.logger.info(f"Client created: {self.client}")

    def logout(self):
        if self.client:
            self.client.logout()
            self.client = None
            self.logger.info("Successfully logged out")

    def get_headers(self):
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    def make_request(
        self, method: str, path: str, params: Dict = None, data: Dict = None
    ):
        url = f"{self.base_url}{path}"
        headers = self.get_headers()

        try:
            response = requests.request(
                method, url, headers=headers, params=params, json=data
            )
            self.logger.debug(f"Request URL: {response.url}")
            self.logger.debug(f"Request headers: {response.request.headers}")
            self.logger.debug(f"Request params: {params}")
            self.logger.debug(f"Request data: {data}")
            self.logger.debug(f"Response status code: {response.status_code}")
            self.logger.debug(f"Response content: {response.text}")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed: {e}")
            if hasattr(e, "response") and e.response is not None:
                self.logger.error(f"Response content: {e.response.text}")
            raise

    def get_position(self) -> int:
        self.logger.info("Retrieving position...")
        # params = {"ticker": self.market_ticker, "settlement_status": "unsettled"}
        positions = self.client.get_positions(ticker=self.market_ticker)
        
        # SDK returns an object; access positions list directly
        positions_list = getattr(positions, "positions", None) or []

        total_position = 0
        for position in positions_list:
            ticker = position.get("ticker") if isinstance(position, dict) else getattr(position, "ticker", None)
            pos_val = position.get("position") if isinstance(position, dict) else getattr(position, "position", 0)
            if ticker == self.market_ticker:
                total_position += int(pos_val)

        self.logger.info(f"Current position: {total_position}")
        return total_position

    def get_price(self) -> Dict[str, float]:
        self.logger.info("Retrieving market data for market ticker: " + self.market_ticker)
        api_response = self.client.get_market(self.market_ticker)
        market_obj = getattr(api_response, "market", None) or {}
        self.logger.info(f"Market object: {getattr(market_obj, 'close_time', None)}")
        yes_bid = float(market_obj.get("yes_bid") if isinstance(market_obj, dict) else getattr(market_obj, "yes_bid", 0)) / 100
        yes_ask = float(market_obj.get("yes_ask") if isinstance(market_obj, dict) else getattr(market_obj, "yes_ask", 0)) / 100
        no_bid = float(market_obj.get("no_bid") if isinstance(market_obj, dict) else getattr(market_obj, "no_bid", 0)) / 100
        no_ask = float(market_obj.get("no_ask") if isinstance(market_obj, dict) else getattr(market_obj, "no_ask", 0)) / 100
        
        yes_mid_price = round((yes_bid + yes_ask) / 2, 2)
        no_mid_price = round((no_bid + no_ask) / 2, 2)

        self.logger.info(f"Current yes mid-market price: ${yes_mid_price:.2f}")
        self.logger.info(f"Current no mid-market price: ${no_mid_price:.2f}")
        return {"yes": yes_mid_price, "no": no_mid_price}

    def get_touch(self):
        m = self.client.get_market(self.market_ticker).market
        def g(obj, k, default=0):
            return (obj[k] if isinstance(obj, dict) else getattr(obj, k, default)) / 100.0
        yes_bid, yes_ask = g(m, "yes_bid"), g(m, "yes_ask")
        no_bid,  no_ask  = g(m, "no_bid"),  g(m, "no_ask")
        return {"yes": (to_tick(yes_bid) if yes_bid else 0.0,
                        to_tick(yes_ask) if yes_ask else 0.0),
                "no":  (to_tick(no_bid)  if no_bid  else 0.0,
                        to_tick(no_ask)  if no_ask  else 0.0)}


    def get_markets(self) -> List[Dict]:
        self.logger.info("Retrieving markets...")
        try:
            cursor = None
            markets: List[Dict] = []
            while True:
                api_response = self.client.get_markets(status='open', cursor=cursor)
                if not api_response:
                    break

                current_markets = getattr(api_response, "markets", None)

                if current_markets:
                    if isinstance(current_markets, list):
                        markets.extend(current_markets)
                    else:
                        markets.append(current_markets)

                self.logger.info(f"Retrieved {len(markets)} markets")
                cursor = getattr(api_response, "cursor", None)
                if not cursor:
                    break
            self.logger.info(f"Retrieved {len(markets)} markets in total")
            # Persist markets to a JSON file for offline inspection
            try:
                with open("markets.json", "w") as f:
                    json.dump(markets, f, indent=2, default=str)
                self.logger.info(f"Wrote {len(markets)} markets to markets.json")
            except Exception as write_error:
                self.logger.error(f"Failed to write markets.json: {write_error}")
            return markets
        except Exception as e:
            self.logger.error(f"Failed to retrieve markets: {e}")
            return []

    def get_markets_by_event(self, event_ticker: str, status: str = 'open') -> List[Dict]:
        self.logger.info(f"Retrieving markets for event {event_ticker}...")
        markets: List[Dict] = []
        try:
            cursor = None
            while True:
                try:
                    api_response = self.client.get_markets(event_ticker=event_ticker, status=status, cursor=cursor)
                except TypeError:
                    # SDK may not support event_ticker filter; fall back to fetching and filtering locally
                    self.logger.warning("SDK get_markets does not accept event_ticker; falling back to local filter")
                    all_markets = self.get_markets()
                    filtered: List[Dict] = []
                    for m in all_markets:
                        et = m.get('event_ticker') if isinstance(m, dict) else getattr(m, 'event_ticker', None)
                        if et == event_ticker:
                            filtered.append(m)
                    self.logger.info(f"Filtered {len(filtered)} markets for event {event_ticker}")
                    return filtered

                if not api_response:
                    break

                current_markets = getattr(api_response, 'markets', None)
                if current_markets:
                    if isinstance(current_markets, list):
                        markets.extend(current_markets)
                    else:
                        markets.append(current_markets)

                cursor = getattr(api_response, 'cursor', None)
                if not cursor:
                    break

            # Normalize to list of dicts
            normalized: List[Dict] = []
            for item in markets:
                if isinstance(item, dict):
                    normalized.append(item)
                    continue
                to_dict_fn = getattr(item, 'to_dict', None)
                if callable(to_dict_fn):
                    try:
                        normalized.append(to_dict_fn())
                        continue
                    except Exception:
                        pass
                model_dump_fn = getattr(item, 'model_dump', None)
                if callable(model_dump_fn):
                    try:
                        normalized.append(model_dump_fn(by_alias=True, exclude_none=True))
                        continue
                    except Exception:
                        pass
                # Fallback minimal fields
                candidate: Dict = {}
                for field in [
                    'ticker', 'event_ticker', 'series_ticker', 'yes_bid', 'yes_ask', 'no_bid', 'no_ask', 'status',
                ]:
                    if hasattr(item, field):
                        candidate[field] = getattr(item, field)
                normalized.append(candidate)

            self.logger.info(f"Retrieved {len(normalized)} markets for event {event_ticker}")
            try:
                with open("markets.json", "w") as f:
                    json.dump(normalized, f, indent=2, default=str)
            except Exception:
                pass
            return normalized
        except Exception as e:
            self.logger.error(f"Failed to retrieve markets for event {event_ticker}: {e}")
            return []

    def get_series(self) -> List[Dict]:
        self.logger.info("Retrieving series...")
        try:
            if not hasattr(self.client, "get_series"):
                self.logger.error("KalshiClient has no method get_series")
                return []

            api_response = self.client.get_series(status='open')
            current_series = getattr(api_response, "series", None)
            if not current_series:
                current_series = []

            for item in current_series:
                self.logger.info(f"Series: {item}")

            try:
                with open("series.json", "w") as f:
                    json.dump(current_series, f, indent=2, default=str)
                self.logger.info(f"Wrote {len(current_series)} series to series.json")
            except Exception as write_error:
                self.logger.error(f"Failed to write series.json: {write_error}")

            self.logger.info(f"Retrieved {len(current_series)} total series")
            return current_series
        except Exception as e:
            self.logger.error(f"Failed to retrieve series: {e}")
            return []

    def place_order(self, action: str, side: str, price: float, quantity: int, expiration_ts: int = None) -> str:
        self.logger.info(f"Placing {action} order for {side} side at price ${price:.2f} with quantity {quantity}...")
        path = "/portfolio/orders"
        data = {
            "ticker": self.market_ticker,
            "action": action.lower(),  # 'buy' or 'sell'
            "type": "limit",
            "side": side,  # 'yes' or 'no'
            "count": quantity,
            "client_order_id": str(uuid.uuid4()),
        }

        price_to_send = max(1, min(99, int(to_cents(price)))) # Convert dollars to cents

        if side == "yes":
            data["yes_price"] = price_to_send
        else:
            data["no_price"] = price_to_send

        if expiration_ts is not None:
            data["expiration_ts"] = expiration_ts

        self.logger.info(f"Data: {data}")
        try:
            # SDK constructs CreateOrderRequest(**kwargs) internally
            response = self.client.create_order(**data)
            # Handle response as model or dict
            order = getattr(response, "order", None)
            if order is None and isinstance(response, dict):
                order = response.get("order")
            order_id = getattr(order, "order_id", None) if order is not None else None
            if order_id is None and isinstance(order, dict):
                order_id = order.get("order_id")
            self.logger.info(f"Placed {action} order for {side} side at price ${price:.2f} with quantity {quantity}, order ID: {order_id}")
            return str(order_id)
        except Exception as e:
            self.logger.error(f"Failed to place order: {e}")
            if hasattr(e, 'response') and e.response is not None:
                self.logger.error(f"Response content: {e.response.text}")
            self.logger.error(f"Request data: {data}")
            raise

    def cancel_order(self, order_id: int) -> bool:
        self.logger.info(f"Canceling order with ID {order_id}...")
        try:
            self.client.cancel_order(order_id)
            self.logger.info(f"Canceled order with ID {order_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def get_orders(self) -> List[Dict]:
        self.logger.info("Retrieving orders...")
        api_response = self.client.get_orders(ticker=self.market_ticker, status="resting")
        raw_orders = getattr(api_response, "orders", None)
        if raw_orders is None:
            # fallback if SDK returns a dict-like response
            raw_orders = api_response.get("orders", []) if isinstance(api_response, dict) else []

        # Normalize to list of dicts so callers can subscript, regardless of SDK model objects
        normalized_orders: List[Dict] = []
        for item in (raw_orders or []):
            if isinstance(item, dict):
                normalized_orders.append(item)
                continue
            # Try SDK model to_dict / model_dump
            to_dict_fn = getattr(item, "to_dict", None)
            if callable(to_dict_fn):
                try:
                    normalized_orders.append(to_dict_fn())
                    continue
                except Exception:
                    pass
            model_dump_fn = getattr(item, "model_dump", None)
            if callable(model_dump_fn):
                try:
                    normalized_orders.append(model_dump_fn(by_alias=True, exclude_none=True))
                    continue
                except Exception:
                    pass
            # Fallback: build a dict from known attributes
            candidate: Dict = {}
            for field in [
                "order_id",
                "client_order_id",
                "ticker",
                "side",
                "action",
                "type",
                "status",
                "yes_price",
                "no_price",
                "count",
                "remaining_count",
                "expiration_time",
                "created_time",
                "updated_time",
            ]:
                if hasattr(item, field):
                    candidate[field] = getattr(item, field)
            normalized_orders.append(candidate)

        self.logger.info(f"Retrieved {len(normalized_orders)} orders")
        return normalized_orders

class AvellanedaMarketMaker:
    def __init__(
        self,
        logger: logging.Logger,
        api: AbstractTradingAPI,
        gamma: float,
        k: float,
        max_position: int,
        order_expiration: int,
        min_spread: float = 0.01,
        position_limit_buffer: float = 0.1,
        inventory_skew_factor: float = 0.01,
        trade_side: str = "yes",
        stop_event: Optional[threading.Event] = None,
        sigma: float = 0.01,
        T: float = 0
    ):
        self.api = api
        self.logger = logger
        self.base_gamma = gamma
        self.k = k
        self.T = T
        self.max_position = max_position
        self.order_expiration = order_expiration
        self.min_spread = min_spread
        self.position_limit_buffer = position_limit_buffer
        self.inventory_skew_factor = inventory_skew_factor
        self.trade_side = trade_side
        self.stop_event = stop_event
        self.sigma = sigma
        # metrics
        self.metrics: Optional[MetricsTracker] = None

    def calc_sigma(self, price: float, logit_prev: float, old_var: float, lambda_: float = 0.96, epsilon: float = 0.0001, sigma_min: float = 0.005, sigma_max: float = 0.05):
        p = max(min(price, 1.0 - epsilon), epsilon)
        logit = math.log(p/(1-p))
        r = logit - logit_prev
        var = lambda_ * old_var + (1 - lambda_) * r**2
        sigma = math.sqrt(var)
        sigma = min(max(sigma, sigma_min), sigma_max)  # clamp
        sigma = price * (1 - price) * sigma
        return sigma, logit, var


    def run(self, dt: float):
        start_time = time.time()
        old_var = 0
        sigma_floor = 0.01        # starting floor in logit-return units
        sigma_min, sigma_max = 0.005, 0.05
        epsilon = 0.0001

        prices = self.api.get_price()
        mid_price = prices[self.trade_side]
        p = max(min(mid_price, 1.0 - epsilon), epsilon)
        logit_prev = math.log(p/(1-p))
        old_var = sigma_floor ** 2   # carry this forward
        sigma = sigma_floor

        tau_seconds = 200
        is_warming_up = True
        secs_to_warm_up = tau_seconds * 3
        secs_in_warm_up = 0


        if self.metrics is None:
            # derive a friendly name from logger
            strategy_name = getattr(self.logger, 'name', 'Strategy')
            self.metrics = MetricsTracker(strategy_name=strategy_name, market_ticker=self.api.market_ticker)
        while (time.time() - start_time < self.T) and (not self._should_stop()):
            prices = self.api.get_price()
            price = prices[self.trade_side]
            lambda_ = math.exp(-dt / tau_seconds)

            sigma_ewma, logit_prev, old_var = self.calc_sigma(price, logit_prev, old_var, lambda_=lambda_, epsilon=epsilon, sigma_min=sigma_min, sigma_max=sigma_max)

            if is_warming_up:
                secs_in_warm_up += dt
                alpha = max(0.0, 1.0 - min(secs_in_warm_up / secs_to_warm_up, 1.0))
                p_curr = max(min(price, 1.0 - epsilon), epsilon)
                sigma_floor_prob = p_curr * (1 - p_curr) * sigma_floor  # convert floor from logit → probability space
                sigma = math.sqrt(alpha * (sigma_floor_prob ** 2) + (1.0 - alpha) * (sigma_ewma ** 2))
                self.logger.info(f"Sigma: {sigma:.4f}")
                if 1 - math.exp(-secs_in_warm_up / tau_seconds) >= 0.90:
                    is_warming_up = False

                # Timing logic - execute regardless of warm-up status
                if self.stop_event is not None:
                    self.stop_event.wait(dt)
                else:
                    time.sleep(dt)
                continue
            else:
                sigma = sigma_ewma

            self.sigma = sigma
            self.logger.info(f"Current sigma: {sigma:.4f}")

            current_time = time.time() - start_time
            self.logger.info(f"Running Avellaneda market maker at {current_time:.2f}")

            t0 = time.time()
            mid_prices = self.api.get_price()
            self.metrics.record_latency("get_price", time.time() - t0)
            mid_price = mid_prices[self.trade_side]
            t0 = time.time()
            inventory = self.api.get_position()
            self.metrics.record_latency("get_position", time.time() - t0)
            self.logger.info(f"Current mid price for {self.trade_side}: {mid_price:.4f}, Inventory: {inventory}")

            reservation_price = self.calculate_reservation_price(mid_price, inventory, current_time)
            self.logger.info(f"Reservation price: {reservation_price:.4f}")
            bid_price, ask_price = self.calculate_asymmetric_quotes(mid_price, inventory, current_time)
            self.logger.info(f"Bid price: {bid_price:.4f}, Ask price: {ask_price:.4f}")
            buy_size, sell_size = self.calculate_order_sizes(inventory)
            self.logger.info(f"Buy size: {buy_size}, Sell size: {sell_size}")

            self.logger.info(f"Computed desired bid: {bid_price:.4f}, ask: {ask_price:.4f}")

            # snapshot
            self.metrics.record_loop(
                t_seconds=current_time,
                mid_price=mid_price,
                inventory=inventory,
                reservation_price=reservation_price,
                bid_price=bid_price,
                ask_price=ask_price,
                buy_size=buy_size,
                sell_size=sell_size,
            )

            self.manage_orders(bid_price, ask_price, buy_size, sell_size)

            # Timing logic for non-warm-up operation
            if self.stop_event is not None:
                self.stop_event.wait(dt)
            else:
                time.sleep(dt)

        self.logger.info("Avellaneda market maker finished running")

    def _should_stop(self) -> bool:
        return bool(self.stop_event.is_set()) if self.stop_event is not None else False

    def export_metrics(self) -> None:
        if self.metrics is None:
            return
        # Safe file prefix based on logger name and ticker
        strategy_name = getattr(self.logger, 'name', 'Strategy')
        safe_name = strategy_name.replace(':', '_').replace(' ', '_')
        base_prefix = f"{safe_name}_{self.api.market_ticker}"
        self.metrics.export_files(base_prefix)

    def calculate_asymmetric_quotes(self, mid_price: float, inventory: int, t: float) -> Tuple[float, float]:
        reservation_price = self.calculate_reservation_price(mid_price, inventory, t)
        self.logger.info(f"Reservation price 2: {reservation_price:.4f}")
        delta = self.calculate_optimal_spread(t, inventory)
        self.logger.info(f"Delta: {delta:.4f}")
        position_ratio = min(abs(inventory) / max(1, self.max_position), 1.0)
        widen_coeff = 0.5
        spread_adjustment = delta * (position_ratio ** 2) * widen_coeff
        self.logger.info(f"Spread adjustment: {spread_adjustment:.4f}")
        if inventory > 0:
            bid_spread = delta + spread_adjustment
            ask_spread = max(delta - 0.5 * spread_adjustment, 0.5 * self.min_spread)
            self.logger.info(f"Bid spread: {bid_spread:.4f}, Ask spread: {ask_spread:.4f}")
        else:
            bid_spread = max(delta - 0.5 * spread_adjustment, 0.5 * self.min_spread)
            ask_spread = delta + spread_adjustment
            self.logger.info(f"Bid spread: {bid_spread:.4f}, Ask spread: {ask_spread:.4f}")
        bid_price = to_tick(max(0.01, min(mid_price, reservation_price - bid_spread)))
        ask_price = to_tick(min(0.99, max(mid_price, reservation_price + ask_spread)))
        self.logger.info(f"Bid price 2: {bid_price:.4f}, Ask price 2: {ask_price:.4f}")
        return bid_price, ask_price

    def calculate_reservation_price(self, mid_price: float, inventory: int, t: float) -> float:
        dynamic_gamma = self.calculate_dynamic_gamma(inventory)
        # inventory_skew = inventory * self.inventory_skew_factor * mid_price
        # return mid_price + inventory_skew - inventory * dynamic_gamma * (self.sigma**2) * (self.T - t)
        return mid_price - inventory * dynamic_gamma * (self.sigma**2) * (self.T - t)

    def calculate_optimal_spread(self, t: float, inventory: int) -> float:
        dynamic_gamma = self.calculate_dynamic_gamma(inventory)
        self.logger.info(f"Dynamic gamma: {dynamic_gamma:.4f}")
        self.logger.info(f"Sigma: {self.sigma:.4f}")
        self.logger.info(f"T: {self.T:.4f}")
        self.logger.info(f"t: {t:.4f}")
        self.logger.info(f"K: {self.k:.4f}")
        base_spread = (dynamic_gamma * (self.sigma**2) * (self.T - t) / 2) + (1 / dynamic_gamma) * math.log(1 + (dynamic_gamma / self.k))
        self.logger.info(f"Base spread: {base_spread:.4f}")
        # position_ratio = abs(inventory) / self.max_position
        # spread_adjustment = 1 + (position_ratio ** 2)
        # return max(base_spread * spread_adjustment * 0.01, self.min_spread)
        return max(base_spread, self.min_spread / 2.0)  # ensure a floor in same price units

    def calculate_dynamic_gamma(self, inventory: int) -> float:
        position_ratio = abs(inventory) / max(1, self.max_position)
        return self.base_gamma * math.exp(position_ratio)   # or: self.base_gamma * (1 + position_ratio)

    def calculate_order_sizes(self, inventory: int) -> Tuple[int, int]:
        # how much more exposure you can add (either direction)
        remaining_capacity = max(0, self.max_position - abs(inventory))
        # a small probing size, bigger on the flattening side
        buffer_size = max(1, int(self.max_position * self.position_limit_buffer))

        if inventory > 0:
            # you’re long → prefer selling more than buying
            sell_size = max(1, min(buffer_size, inventory))             # can’t sell more than you hold
            buy_size  = max(1, min(buffer_size, remaining_capacity))    # still allow small buys for spread capture
        elif inventory < 0:
            # you’re short → prefer buying more than selling
            buy_size  = max(1, min(buffer_size, remaining_capacity))
            sell_size = max(1, min(buffer_size, abs(inventory)))        # can’t add to short too aggressively
        else:
            # flat → symmetric small sizes
            buy_size = sell_size = max(1, min(buffer_size, remaining_capacity or buffer_size))
        return buy_size, sell_size


    def manage_orders(self, bid_price: float, ask_price: float, buy_size: int, sell_size: int):
        t0 = time.time()
        current_orders = self.api.get_orders()
        if self.metrics:
            self.metrics.record_latency("get_orders", time.time() - t0)
        self.logger.info(f"Retrieved {len(current_orders)} total orders")

        buy_orders = []
        sell_orders = []

        self.logger.info(f"Current orders: {current_orders}")
        for order in current_orders:
            self.logger.info(f"Order: {order}")
            if order['side'] == self.trade_side:
                if order['action'] == 'buy':
                    buy_orders.append(order)
                elif order['action'] == 'sell':
                    sell_orders.append(order)

        self.logger.info(f"Current buy orders: {len(buy_orders)}")
        self.logger.info(f"Current sell orders: {len(sell_orders)}")

        # Handle buy orders
        self.handle_order_side('buy', buy_orders, bid_price, buy_size)

        # Handle sell orders
        self.handle_order_side('sell', sell_orders, ask_price, sell_size)

    def handle_order_side(self, action: str, orders: List[Dict], desired_price: float, desired_size: int):
        desired_price = to_tick(desired_price)
        desired_size = max(1, desired_size)

        touch_bid, touch_ask = self.api.get_touch()[self.trade_side]
        spread = (touch_ask - touch_bid) if (touch_bid and touch_ask) else None
        self.logger.info(f"Touch {self.trade_side}: bid={touch_bid:.2f} ask={touch_ask:.2f} | target={desired_price:.2f}")

        target = desired_price
        if action == "buy":
            # Join best bid at least
            if touch_bid:
                target = max(target, touch_bid)
            # Optional: improve by 1 tick only if there is room (kept simple; you can add conditions)
            if spread is not None and spread >= 0.02 and touch_ask:
                target = min(target + 0.01, touch_ask - 0.01)
        else:  # sell
            # Join best ask at least
            if touch_ask:
                target = min(target, touch_ask)
            if spread is not None and spread >= 0.02 and touch_bid:
                target = max(target - 0.01, touch_bid + 0.01)

        target = to_tick(target)
        keep_order = None
        for order in orders:
            px = (float(order["yes_price"]) if self.trade_side == "yes"
                else float(order["no_price"])) / 100.0
            px = to_tick(px)
            if keep_order is None and px == target:
                keep_order = order
                self.logger.info(f"Keeping existing {action} at {px:.2f} (order_id={order['order_id']})")
                if self.metrics:
                    self.metrics.record_action("keep_order", {
                        "order_id": order["order_id"], "action": action,
                        "side": self.trade_side, "price": px, "size": order.get("remaining_count","")})
            else:
                # Only cancel if price tick differs; keep partials to preserve queue
                if px != target:
                    self.logger.info(f"Cancelling {action} at {px:.2f} (want {target:.2f}) id={order['order_id']}")
                    t0 = time.time()
                    self.api.cancel_order(order["order_id"])
                    if self.metrics:
                        self.metrics.record_latency("cancel_order", time.time() - t0)
                        self.metrics.record_action("cancel_order", {
                            "order_id": order["order_id"], "action": action,
                            "side": self.trade_side, "price": px, "size": order.get("remaining_count","")})

        # 5) If nothing to keep, PLACE at target tick — no gating on mid or stale current_price
        if keep_order is None:
            try:
                t0 = time.time()
                order_id = self.api.place_order(
                    action, self.trade_side, target, desired_size,
                    int(time.time()) + self.order_expiration
                )
                if self.metrics:
                    self.metrics.record_latency("place_order", time.time() - t0)
                    self.metrics.record_action("place_order", {
                        "order_id": order_id, "action": action, "side": self.trade_side,
                        "price": target, "size": desired_size})
                self.logger.info(f"Placed {action} {desired_size}@{target:.2f} (order_id={order_id})")
            except Exception as e:
                self.logger.error(f"Failed to place {action}: {e}")

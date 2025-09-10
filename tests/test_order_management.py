import time
import pytest


def cents(p):
    return int(round(p * 100))


def make_order(action, side, price, qty, order_id='1'):
    key = 'yes_price' if side == 'yes' else 'no_price'
    return {
        'order_id': order_id,
        'ticker': 'TEST',
        'action': action,
        'side': side,
        'type': 'limit',
        key: cents(price),
        'count': qty,
        'remaining_count': qty,
        'status': 'resting',
    }


def test_handle_order_side_keeps_matching_order(maker, dummy_api):
    # Current mid for yes
    dummy_api.set_price(yes=0.60)
    # Desired bid is 0.58 with qty 2
    desired_price = 0.58
    desired_qty = 2
    existing = make_order('buy', 'yes', desired_price, desired_qty, order_id='keep-me')
    dummy_api.set_orders([existing])

    maker.handle_order_side('buy', dummy_api.get_orders(), desired_price, desired_qty)

    # No cancellations and no new orders placed
    assert dummy_api.canceled_orders == []
    assert not any(o['order_id'] != 'keep-me' for o in dummy_api.get_orders())


def test_handle_order_side_cancels_extraneous_orders(maker, dummy_api):
    dummy_api.set_price(yes=0.60)
    desired_price = 0.58
    desired_qty = 2
    keep = make_order('buy', 'yes', desired_price, desired_qty, order_id='keep')
    worse = make_order('buy', 'yes', 0.55, desired_qty, order_id='cancel1')
    wrong_qty = make_order('buy', 'yes', desired_price, 5, order_id='cancel2')
    dummy_api.set_orders([keep, worse, wrong_qty])

    maker.handle_order_side('buy', dummy_api.get_orders(), desired_price, desired_qty)

    # Extraneous orders are cancelled
    assert set(dummy_api.canceled_orders) == {'cancel1', 'cancel2'}
    # Keep order remains
    ids = {o['order_id'] for o in dummy_api.get_orders()}
    assert 'keep' in ids and 'cancel1' not in ids and 'cancel2' not in ids


def test_handle_order_side_places_if_improving(maker, dummy_api):
    # For a buy, desired price must be below current price to improve
    dummy_api.set_price(yes=0.60)
    desired_price = 0.58
    desired_qty = 3
    dummy_api.set_orders([])

    maker.handle_order_side('buy', [], desired_price, desired_qty)

    assert len(dummy_api.placed_orders) == 1
    placed = dummy_api.placed_orders[0]
    assert placed['action'] == 'buy'
    assert placed['side'] == 'yes'
    assert placed['remaining_count'] == desired_qty
    assert placed['yes_price'] == cents(desired_price)


def test_handle_order_side_skips_if_not_improving(maker, dummy_api):
    dummy_api.set_price(yes=0.55)
    desired_price = 0.56  # worse than current for a buy
    desired_qty = 1
    dummy_api.set_orders([])

    maker.handle_order_side('buy', [], desired_price, desired_qty)

    assert len(dummy_api.placed_orders) == 0


def test_manage_orders_splits_by_side_and_calls_handle(maker, dummy_api, monkeypatch):
    # Set some existing orders of both actions
    dummy_api.set_orders([
        make_order('buy', 'yes', 0.51, 1, 'b1'),
        make_order('sell', 'yes', 0.59, 1, 's1'),
    ])

    calls = []

    def spy(action, orders, price, size):
        calls.append((action, [o['order_id'] for o in orders], price, size))

    monkeypatch.setattr(maker, 'handle_order_side', spy)

    maker.manage_orders(bid_price=0.50, ask_price=0.60, buy_size=2, sell_size=3)

    assert ('buy', ['b1'], 0.50, 2) in calls
    assert ('sell', ['s1'], 0.60, 3) in calls



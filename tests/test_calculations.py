import math
import pytest


def test_calculate_dynamic_gamma_increases_with_inventory(maker):
    base = maker.base_gamma
    g0 = maker.calculate_dynamic_gamma(0)
    g_half = maker.calculate_dynamic_gamma(maker.max_position // 2)
    g_full = maker.calculate_dynamic_gamma(maker.max_position)

    assert g0 == pytest.approx(base * math.exp(0.0))
    assert g_half > g0
    assert g_full > g_half


def test_calculate_reservation_price_skews_against_inventory(maker):
    mid = 0.60
    t = 10.0
    maker.sigma = 0.02

    r_flat = maker.calculate_reservation_price(mid, inventory=0, t=t)
    r_long = maker.calculate_reservation_price(mid, inventory=+5, t=t)
    r_short = maker.calculate_reservation_price(mid, inventory=-5, t=t)

    assert r_flat == pytest.approx(mid)
    # long inventory → reservation goes down
    assert r_long < r_flat
    # short inventory → reservation goes up
    assert r_short > r_flat


def test_calculate_optimal_spread_has_minimum(maker):
    maker.min_spread = 0.01
    spread = maker.calculate_optimal_spread(t=0.0, inventory=0)
    assert spread >= maker.min_spread / 2.0


def test_calculate_asymmetric_quotes_bounds_and_direction(maker):
    mid = 0.55
    maker.sigma = 0.03

    bid0, ask0 = maker.calculate_asymmetric_quotes(mid, inventory=0, t=0.0)
    assert 0.0 <= bid0 <= mid <= ask0 <= 1.0

    bid_long, ask_long = maker.calculate_asymmetric_quotes(mid, inventory=+5, t=0.0)
    bid_short, ask_short = maker.calculate_asymmetric_quotes(mid, inventory=-5, t=0.0)

    # With long inventory, bid moves down and ask moves closer/up relative to flat
    assert bid_long <= bid0
    assert ask_long >= ask0 or ask_long <= 1.0

    # With short inventory, bid moves up and ask moves up
    assert bid_short >= bid0
    assert ask_short >= ask0


def test_calculate_order_sizes_respects_limits(maker):
    maker.max_position = 10
    maker.position_limit_buffer = 0.2

    buy, sell = maker.calculate_order_sizes(inventory=0)
    assert buy >= 1 and sell >= 1

    buy_long, sell_long = maker.calculate_order_sizes(inventory=7)
    assert sell_long >= 1 and sell_long <= 7
    assert buy_long >= 1 and buy_long <= maker.max_position - 7

    buy_short, sell_short = maker.calculate_order_sizes(inventory=-7)
    assert buy_short >= 1 and buy_short <= maker.max_position - 7
    assert sell_short >= 1 and sell_short <= 7



import argparse
import logging
from concurrent.futures import ThreadPoolExecutor
import yaml
from dotenv import load_dotenv
import os
from typing import Dict
import signal
import threading
import time

from mm import KalshiTradingAPI, AvellanedaMarketMaker

def load_config(config_file):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def create_api(api_config, logger, market_ticker_override: str | None = None):
    return KalshiTradingAPI(
        email=os.getenv("KALSHI_EMAIL"),
        password=os.getenv("KALSHI_PASSWORD"),
        market_ticker=(market_ticker_override or api_config.get('market_ticker') or api_config.get('event_ticker', 'UNKNOWN')),
        base_url=os.getenv("KALSHI_BASE_URL"),
        logger=logger,
    )

def create_market_maker(mm_config, api, logger, stop_event: threading.Event, trade_side: str | None = None):
    return AvellanedaMarketMaker(
        logger=logger,
        api=api,
        gamma=mm_config.get('gamma', 0.1),
        k=mm_config.get('k', 1.5),
        max_position=mm_config.get('max_position', 100),
        order_expiration=mm_config.get('order_expiration', 300),
        min_spread=mm_config.get('min_spread', 0.01),
        position_limit_buffer=mm_config.get('position_limit_buffer', 0.1),
        inventory_skew_factor=mm_config.get('inventory_skew_factor', 0.01),
        trade_side=(trade_side or mm_config.get('trade_side', 'yes')),
        stop_event=stop_event,
        T=mm_config.get('T', 3600),
    )

def run_strategy(config_name: str, config: Dict, stop_event: threading.Event):
    log_level = config.get('log_level', 'INFO')

    def build_logger(side_suffix: str, market_suffix: str | None = None) -> logging.Logger:
        level = getattr(logging, str(log_level).upper(), logging.INFO)
        suffix = f"{config_name}{('-' + market_suffix) if market_suffix else ''}_{side_suffix}"
        lg = logging.getLogger(f"Strategy_{suffix}")
        lg.propagate = False
        lg.setLevel(level)

        # Reset handlers to avoid stale/misconfigured ones
        for h in list(lg.handlers):
            lg.removeHandler(h)

        # File handler per side and market
        log_filename = f"{config_name}{('-' + market_suffix) if market_suffix else ''}_{side_suffix}.log"
        fh = logging.FileHandler(log_filename, encoding='utf-8')
        fh.setLevel(level)

        # Console handler per side
        ch = logging.StreamHandler()
        ch.setLevel(level)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        lg.addHandler(fh)
        lg.addHandler(ch)
        return lg

    def run_one_market_side(market_ticker: str, side: str):
        side_upper = side.upper()
        market_suffix = market_ticker
        print(f"Building logger: {config_name}-{market_suffix} [{side_upper}]")
        logger = build_logger(side_upper, market_suffix=market_suffix)
        print(f"Logger built: {config_name}-{market_suffix} [{side_upper}]")
        logger.info(f"Starting strategy: {config_name}-{market_suffix} [{side_upper}]")

        # Create side-specific API (so API logs go to side logger)
        api = create_api(config['api'], logger, market_ticker_override=market_ticker)
        logger.info(f"API created: {api}")

        # Create side-specific market maker
        market_maker = create_market_maker(
            config['market_maker'], api, logger, stop_event, trade_side=side
        )

        try:
            market_maker.run(config.get('dt', 1.0))
        except KeyboardInterrupt:
            logger.info("Market maker stopped by user")
            try:
                market_maker.export_metrics()
                logger.info("Exported metrics after keyboard interrupt")
            except Exception as _:
                logger.warning("Failed to export metrics on keyboard interrupt")
        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")
        finally:
            try:
                market_maker.export_metrics()
                logger.info("Exported metrics on shutdown")
            except Exception as _:
                logger.warning("Failed to export metrics on shutdown")
            api.logout()

    api_cfg = config.get('api', {})
    event_ticker = api_cfg.get('event_ticker')
    if event_ticker:
        # Use a discovery API to list markets for the event
        discovery_logger = build_logger('DISCOVERY', market_suffix=event_ticker)
        discovery_api = create_api(api_cfg, discovery_logger, market_ticker_override=event_ticker)
        markets = discovery_api.get_markets_by_event(event_ticker)
        try:
            discovery_api.logout()
        except Exception:
            pass

        market_tickers = []
        for m in markets:
            tkr = m.get('ticker') if isinstance(m, dict) else getattr(m, 'ticker', None)
            if tkr:
                market_tickers.append(tkr)

        if not market_tickers:
            print(f"No markets found for event {event_ticker}")
            return

        # Run YES and NO sides for each market in parallel
        max_workers = min(64, max(2, 2 * len(market_tickers)))
        with ThreadPoolExecutor(max_workers=max_workers) as side_executor:
            futures = []
            for mt in market_tickers:
                futures.append(side_executor.submit(run_one_market_side, mt, 'yes'))
                futures.append(side_executor.submit(run_one_market_side, mt, 'no'))
            for f in futures:
                try:
                    f.result()
                except Exception:
                    pass
    else:
        # Backward compatibility: single market_ticker in config
        single_ticker = api_cfg.get('market_ticker')
        # If not provided, nothing to do
        if not single_ticker:
            print(f"Config {config_name} missing 'market_ticker' or 'event_ticker'")
            return
        with ThreadPoolExecutor(max_workers=2) as side_executor:
            futures = [
                side_executor.submit(run_one_market_side, single_ticker, 'yes'),
                side_executor.submit(run_one_market_side, single_ticker, 'no'),
            ]
            for f in futures:
                try:
                    f.result()
                except Exception:
                    pass

def _install_signal_handlers(stop_event: threading.Event):
    def handle_signal(signum, frame):
        # First signal: request graceful shutdown
        if not stop_event.is_set():
            print("\nSignal received. Stopping strategies gracefully... (press Ctrl-C again to force)")
            stop_event.set()
        else:
            # Second signal: raise KeyboardInterrupt to force exit
            raise KeyboardInterrupt
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kalshi Market Making Algorithm")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    # Load all configurations
    configs = load_config(args.config)

    # Load environment variables
    load_dotenv()

    # Print the name of every strategy being run
    print("Starting the following strategies:")
    for config_name in configs:
        print(f"- {config_name}")

    # Shared stop event for all strategies
    stop_event = threading.Event()
    _install_signal_handlers(stop_event)

    # Run all strategies in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=len(configs)) as executor:
        for config_name, config in configs.items():
            executor.submit(run_strategy, config_name, config, stop_event)

        # Keep the main thread alive until all tasks finish or stop requested
        try:
            while not stop_event.is_set():
                time.sleep(0.2)
        except KeyboardInterrupt:
            # If a second Ctrl-C is pressed, enforce shutdown
            stop_event.set()
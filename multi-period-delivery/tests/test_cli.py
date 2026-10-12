from multi_period_delivery import ALNSConfig, DataALNS, run_model
from multi_period_delivery.alns import ALNSConfig as SubpackageALNSConfig
from multi_period_delivery.cli import build_parser


def test_cli_parser_defaults_to_small_alns_b() -> None:
    parser = build_parser()
    args = parser.parse_args([])

    assert args.scale == "small"
    assert args.dataset_id == 1
    assert args.enable_param_tuner is False
    assert args.enable_ml_operators is False


def test_alns_public_imports_are_preserved() -> None:
    assert ALNSConfig is SubpackageALNSConfig
    assert DataALNS.__name__ == "DataALNS"
    assert callable(run_model)

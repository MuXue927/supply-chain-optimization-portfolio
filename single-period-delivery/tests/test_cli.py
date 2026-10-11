from single_period_delivery.cli import build_parser


def test_cli_parser_has_expected_algorithms() -> None:
    parser = build_parser()
    args = parser.parse_args(["heuristic", "--experiments", "2"])

    assert args.algorithm == "heuristic"
    assert args.experiments == 2
    assert args.dataset_name == "dataset_9"

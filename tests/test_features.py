import unittest

from smartsignal.data import generate_demo_data
from smartsignal.features import (
    FORWARD_RETURN_COLUMN,
    TARGET_COLUMN,
    build_feature_frame,
    feature_columns,
)


class FeatureTests(unittest.TestCase):
    def test_target_matches_next_day_return(self) -> None:
        market = generate_demo_data(periods=500)
        frame = build_feature_frame(market)
        self.assertTrue(
            (
                frame[TARGET_COLUMN]
                == (frame[FORWARD_RETURN_COLUMN] > 0.0).astype(int)
            ).all()
        )

    def test_future_price_change_does_not_change_prior_features(self) -> None:
        market = generate_demo_data(periods=500)
        original = build_feature_frame(market, include_unlabeled=True)
        changed = market.copy()
        changed.iloc[-1, changed.columns.get_loc("close")] *= 2.0
        recalculated = build_feature_frame(changed, include_unlabeled=True)

        comparison_date = original.index[-2]
        for column in feature_columns(original):
            self.assertAlmostEqual(
                original.loc[comparison_date, column],
                recalculated.loc[comparison_date, column],
            )

    def test_unlabeled_frame_retains_latest_day(self) -> None:
        market = generate_demo_data(periods=500)
        frame = build_feature_frame(market, include_unlabeled=True)
        self.assertEqual(frame.index[-1], market.index[-1])
        self.assertTrue(frame[TARGET_COLUMN].isna().iloc[-1])


if __name__ == "__main__":
    unittest.main()


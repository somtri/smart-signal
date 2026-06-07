import unittest

import pandas as pd

from smartsignal.data import attach_daily_sentiment, generate_demo_data


class DataTests(unittest.TestCase):
    def test_demo_data_is_deterministic(self) -> None:
        first = generate_demo_data(periods=450, seed=7)
        second = generate_demo_data(periods=450, seed=7)
        pd.testing.assert_frame_equal(first, second)

    def test_daily_sentiment_preserves_headline_counts(self) -> None:
        market = generate_demo_data(periods=450).drop(
            columns=["sentiment", "headline_count"]
        )
        date = market.index[10]
        sentiment = pd.DataFrame(
            {
                "date": [date, date],
                "sentiment": [0.8, 0.2],
                "headline_count": [3, 2],
            }
        )
        joined = attach_daily_sentiment(market, sentiment)
        self.assertAlmostEqual(joined.loc[date, "sentiment"], 0.5)
        self.assertEqual(joined.loc[date, "headline_count"], 5)


if __name__ == "__main__":
    unittest.main()


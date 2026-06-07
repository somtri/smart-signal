import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("streamlit"), "streamlit is not installed")
class DashboardTests(unittest.TestCase):
    def test_dashboard_renders_without_exceptions(self) -> None:
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file("app.py").run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.title[0].value, "SmartSignal")
        self.assertEqual(len(app.metric), 5)
        self.assertEqual(len(app.get("plotly_chart")), 3)


if __name__ == "__main__":
    unittest.main()


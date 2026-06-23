import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


class AppSmokeTests(unittest.TestCase):
    def test_default_and_two_variable_runs_render_without_exceptions(self):
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        at = AppTest.from_file(str(app_path), default_timeout=120)
        at.run(timeout=120)
        self.assertEqual(len(at.exception), 0, [element.value for element in at.exception])
        self.assertEqual(len(at.tabs), 6)

        at.sidebar.multiselect[0].set_value(["Raf", "Mek"])
        at.run(timeout=120)
        self.assertEqual(len(at.exception), 0, [element.value for element in at.exception])


if __name__ == "__main__":
    unittest.main()

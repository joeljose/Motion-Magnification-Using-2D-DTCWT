"""Unit tests for motion_mag.py — Phase-Based Motion Magnification."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import motion_mag


class TestFormatDuration:
    def test_seconds_only(self):
        assert motion_mag.format_duration(30.0) == "30.0s"

    def test_minutes_and_seconds(self):
        assert motion_mag.format_duration(90.5) == "1m 30.5s"

    def test_zero(self):
        assert motion_mag.format_duration(0) == "0.0s"

    def test_exactly_60(self):
        assert motion_mag.format_duration(60.0) == "1m 0.0s"

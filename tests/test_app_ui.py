"""The Streamlit script runs end to end and renders the reference panels.

``AppTest`` executes app.py in-process, so a rendering error in any panel fails
here rather than in the browser.
"""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from it_support.config import PROJECT_ROOT
from it_support.ui_theme import priority_chip, status_chip

APP = str(PROJECT_ROOT / "app.py")

pytestmark = pytest.mark.usefixtures("temporary_database")


def run_app() -> AppTest:
    app = AppTest.from_file(APP, default_timeout=60)
    app.run()
    return app


def test_the_app_renders_without_an_exception():
    app = run_app()
    assert not app.exception


def test_the_reference_pane_has_all_three_sections():
    app = run_app()
    labels = [tab.label for tab in app.tabs]
    assert labels == ["📚 Knowledge base", "🎫 Tickets", "👥 Directory & status"]


def test_the_ticket_table_shows_the_seeded_tickets():
    app = run_app()
    frames = app.dataframe
    assert frames, "expected the ticket and directory tables to be rendered"
    tickets = frames[0].value
    assert len(tickets) == 18
    assert "INC-10001" in tickets["Ticket"].tolist()


def test_the_identity_banner_starts_anonymous():
    app = run_app()
    markdown = " ".join(element.value for element in app.markdown)
    assert "No employee identified yet" in markdown


def test_sample_prompt_buttons_are_offered():
    app = run_app()
    labels = [button.label for button in app.button]
    assert "How do I reset my VPN password?" in labels
    assert "🧹 Clear conversation" in labels


def test_status_and_priority_chips_carry_their_own_colour():
    assert "#b42318" in status_chip("Open")
    assert "#067647" in status_chip("Resolved")
    assert "#b42318" in priority_chip("Critical")

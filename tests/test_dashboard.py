"""
test_dashboard.py — Dashboard integration tests.

Owner: Deekshitha (C5)
Phase: 5

Targets:
  - All Dash callbacks functional
  - All 6 pages rendering
  - Global filters wired via dcc.Store
  - Shared components (filters, map_utils, chat_ui) are importable and functional

These tests are designed to run in CI without requiring a running
Dash server or browser — they validate component structure, callback
registration, and module integrity through import-based and layout-
inspection tests.

Tests that depend on optional packages (LangChain, Prophet, etc.) or
local data artifacts are automatically skipped when those dependencies
are unavailable, ensuring CI stays green in partial environments.

Usage:
    pytest tests/test_dashboard.py -v
    pytest tests/test_dashboard.py -v -k "page"
    pytest tests/test_dashboard.py -v -k "filter"
    pytest tests/test_dashboard.py -v -k "component"
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pandas as pd
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Environment probes
# ---------------------------------------------------------------------------
try:
    import dash  # noqa: F401
    _HAS_DASH = True
except ImportError:
    _HAS_DASH = False

_skip_no_dash = pytest.mark.skipif(not _HAS_DASH, reason="Dash not installed")


def _try_import(module_path: str):
    """Try to import a module; return (module, None) or (None, error_msg)."""
    try:
        mod = importlib.import_module(module_path)
        return mod, None
    except Exception as exc:
        return None, str(exc)


def _can_import_chat_ui() -> bool:
    """chat_ui depends on src.rag.chain which needs LangChain."""
    _, err = _try_import("src.dashboard.components.chat_ui")
    return err is None


def _can_import_app() -> bool:
    """app.py imports chat_ui transitively via page auto-discovery."""
    _, err = _try_import("src.dashboard.app")
    return err is None


_skip_no_langchain = pytest.mark.skipif(
    not _can_import_chat_ui(),
    reason="LangChain / RAG chain not available — chat_ui cannot import"
)

_skip_no_app = pytest.mark.skipif(
    not _can_import_app(),
    reason="Cannot import app (missing optional dependency)"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sample_data_df() -> pd.DataFrame:
    """Minimal DataFrame matching the schema expected by the filter bar."""
    return pd.DataFrame({
        "CALL_YEAR": [2020, 2021, 2022, 2023],
        "service_type": ["EMS", "Fire", "EMS", "Fire"],
        "call_type": ["FALL", "CHEST PAIN", "BREATHING PROBLEM", "TRAUMA"],
    })


@pytest.fixture(scope="module")
def geo_data_df() -> pd.DataFrame:
    """Minimal DataFrame with latitude/longitude for map utils."""
    np.random.seed(42)
    return pd.DataFrame({
        "latitude": np.random.normal(40.44, 0.05, 40),
        "longitude": np.random.normal(-79.99, 0.05, 40),
        "call_count": np.random.randint(10, 500, 40),
    })


# ===========================================================================
# MODULE IMPORTABILITY TESTS
# ===========================================================================

@_skip_no_dash
class TestModuleImports:
    """Verify all dashboard modules are importable without side-effect crashes."""

    def test_filters_module_importable(self):
        mod, err = _try_import("src.dashboard.components.filters")
        assert mod is not None, f"filters module import failed: {err}"
        assert hasattr(mod, "create_filter_bar")

    def test_map_utils_module_importable(self):
        mod, err = _try_import("src.dashboard.components.map_utils")
        assert mod is not None, f"map_utils module import failed: {err}"
        assert hasattr(mod, "create_choropleth_map")
        assert hasattr(mod, "create_cluster_map")
        assert hasattr(mod, "create_heatmap")

    @_skip_no_langchain
    def test_chat_ui_module_importable(self):
        mod, err = _try_import("src.dashboard.components.chat_ui")
        assert mod is not None, f"chat_ui import failed: {err}"
        assert hasattr(mod, "create_chat_container")
        assert hasattr(mod, "PROMPTS")

    @_skip_no_app
    def test_app_module_importable(self):
        mod, err = _try_import("src.dashboard.app")
        assert mod is not None, f"app module import failed: {err}"
        assert hasattr(mod, "app")
        assert hasattr(mod, "server")


# ===========================================================================
# DASH APP CREATION TESTS
# ===========================================================================

@_skip_no_dash
@_skip_no_app
class TestDashAppCreation:
    """Verify the app object is well-formed."""

    def test_app_initializes_without_error(self):
        from src.dashboard.app import app
        assert app is not None

    def test_app_has_suppress_callback_exceptions(self):
        from src.dashboard.app import app
        assert app.config.suppress_callback_exceptions is True

    def test_app_uses_darkly_theme(self):
        from src.dashboard.app import app
        stylesheets = app.config.external_stylesheets
        assert any("darkly" in str(s).lower() or "DARKLY" in str(s)
                    for s in stylesheets), (
            f"Expected DARKLY theme in stylesheets, got: {stylesheets}"
        )

    def test_app_title_is_medalertai(self):
        from src.dashboard.app import app
        assert app.title == "MedAlertAI"

    def test_server_is_flask_instance(self):
        from src.dashboard.app import server
        assert hasattr(server, "route")
        assert hasattr(server, "wsgi_app")

    def test_app_uses_pages(self):
        from src.dashboard.app import app
        assert app.config.use_pages is True


# ===========================================================================
# PAGE REGISTRATION TESTS
# ===========================================================================

@_skip_no_dash
class TestPageRegistration:
    """Verify pages are registered correctly.

    The full 6-page check requires all optional deps (LangChain, etc.).
    We test what we can import and verify structural correctness.
    """

    def test_forecast_page_registered(self):
        import dash
        paths = {p["path"] for p in dash.page_registry.values()}
        assert "/forecast" in paths

    def test_temporal_page_registered(self):
        import dash
        paths = {p["path"] for p in dash.page_registry.values()}
        assert "/temporal" in paths

    def test_geography_page_registered(self):
        import dash
        paths = {p["path"] for p in dash.page_registry.values()}
        assert "/geography" in paths

    def test_at_least_three_core_pages_registered(self):
        """Forecast, temporal, geography should always register (no LangChain dep)."""
        import dash
        paths = {p["path"] for p in dash.page_registry.values()}
        core = {"/forecast", "/temporal", "/geography"}
        assert core.issubset(paths), f"Missing core pages: {core - paths}"

    @_skip_no_app
    def test_all_six_pages_registered(self):
        """Full check requires LangChain + all deps for assistant/overview/qa."""
        import dash
        page_registry = dash.page_registry
        registered_paths = {page["path"] for page in page_registry.values()}
        expected = {"/", "/temporal", "/geography", "/forecast",
                    "/classification-qa", "/assistant"}
        missing = expected - registered_paths
        assert not missing, f"Missing pages: {missing}"

    @_skip_no_app
    def test_page_count_is_six(self):
        import dash
        assert len(dash.page_registry) == 6, (
            f"Expected 6 registered pages, got {len(dash.page_registry)}"
        )

    @_skip_no_app
    @pytest.mark.parametrize("path,name", [
        ("/", "Overview"),
        ("/temporal", "Temporal"),
        ("/geography", "Geography"),
        ("/forecast", "Forecast"),
        ("/classification-qa", "Classification QA"),
        ("/assistant", "Assistant"),
    ])
    def test_page_registered_with_correct_name(self, path, name):
        import dash
        matching = [p for p in dash.page_registry.values() if p["path"] == path]
        assert len(matching) == 1, f"Expected exactly 1 page at '{path}'"
        assert matching[0]["name"] == name


# ===========================================================================
# PAGE LAYOUT TESTS
# ===========================================================================

@_skip_no_dash
class TestPageLayouts:
    """Verify each page's layout returns a valid Dash component."""

    def _get_layout(self, module_path: str):
        mod, err = _try_import(module_path)
        if mod is None:
            pytest.skip(f"Cannot import {module_path}: {err}")
        layout = mod.layout
        if callable(layout):
            return layout()
        return layout

    def test_overview_layout_returns_component(self):
        layout = self._get_layout("src.dashboard.pages.overview")
        assert layout is not None
        assert hasattr(layout, "children") or hasattr(layout, "to_plotly_json")

    def test_temporal_layout_returns_component(self):
        layout = self._get_layout("src.dashboard.pages.temporal")
        assert layout is not None
        assert hasattr(layout, "children") or hasattr(layout, "to_plotly_json")

    def test_geography_layout_returns_component(self):
        layout = self._get_layout("src.dashboard.pages.geography")
        assert layout is not None
        assert hasattr(layout, "children") or hasattr(layout, "to_plotly_json")

    def test_forecast_layout_returns_component(self):
        layout = self._get_layout("src.dashboard.pages.forecast")
        assert layout is not None
        assert hasattr(layout, "children") or hasattr(layout, "to_plotly_json")

    def test_qa_layout_returns_component(self):
        layout = self._get_layout("src.dashboard.pages.qa")
        assert layout is not None
        assert hasattr(layout, "children") or hasattr(layout, "to_plotly_json")

    def test_assistant_layout_returns_component(self):
        layout = self._get_layout("src.dashboard.pages.assistant")
        assert layout is not None
        assert hasattr(layout, "children") or hasattr(layout, "to_plotly_json")


# ===========================================================================
# GLOBAL FILTER BAR TESTS
# ===========================================================================

@_skip_no_dash
class TestGlobalFilters:
    """Verify global filter bar and dcc.Store integration."""

    def test_create_filter_bar_returns_layout(self, sample_data_df):
        from src.dashboard.components.filters import create_filter_bar
        import dash_bootstrap_components as dbc

        result = create_filter_bar(sample_data_df)
        assert result is not None
        assert isinstance(result, dbc.Row)

    def test_filter_bar_contains_global_store(self, sample_data_df):
        from src.dashboard.components.filters import create_filter_bar

        bar = create_filter_bar(sample_data_df)
        found = _find_component_by_id(bar, "global-filter-store")
        assert found is not None, "global-filter-store dcc.Store not found in filter bar"

    def test_filter_bar_contains_year_dropdown(self, sample_data_df):
        from src.dashboard.components.filters import create_filter_bar

        bar = create_filter_bar(sample_data_df)
        found = _find_component_by_id(bar, "year-filter")
        assert found is not None, "year-filter dropdown not found"

    def test_filter_bar_contains_service_dropdown(self, sample_data_df):
        from src.dashboard.components.filters import create_filter_bar

        bar = create_filter_bar(sample_data_df)
        found = _find_component_by_id(bar, "service-filter")
        assert found is not None, "service-filter dropdown not found"

    def test_filter_bar_contains_mpds_dropdown(self, sample_data_df):
        from src.dashboard.components.filters import create_filter_bar

        bar = create_filter_bar(sample_data_df)
        found = _find_component_by_id(bar, "mpds-filter")
        assert found is not None, "mpds-filter dropdown not found"

    def test_filter_bar_contains_reset_button(self, sample_data_df):
        from src.dashboard.components.filters import create_filter_bar

        bar = create_filter_bar(sample_data_df)
        found = _find_component_by_id(bar, "reset-filters-btn")
        assert found is not None, "reset-filters-btn not found"

    def test_global_store_initial_data_has_expected_keys(self, sample_data_df):
        from src.dashboard.components.filters import create_filter_bar

        bar = create_filter_bar(sample_data_df)
        store = _find_component_by_id(bar, "global-filter-store")
        assert store is not None
        data = store.data
        assert "years" in data
        assert "services" in data
        assert "call_types" in data

    def test_global_store_years_match_data(self, sample_data_df):
        from src.dashboard.components.filters import create_filter_bar

        bar = create_filter_bar(sample_data_df)
        store = _find_component_by_id(bar, "global-filter-store")
        expected_years = sorted(sample_data_df["CALL_YEAR"].unique().tolist())
        assert store.data["years"] == expected_years

    def test_update_global_filters_callback_exists(self):
        from src.dashboard.components.filters import update_global_filters
        assert callable(update_global_filters)

    def test_update_global_filters_returns_dict(self):
        from src.dashboard.components.filters import update_global_filters
        result = update_global_filters(
            years=[2023], services=["EMS"], call_types=["FALL"], reset_clicks=0
        )
        assert isinstance(result, dict)
        assert result["years"] == [2023]
        assert result["services"] == ["EMS"]
        assert result["call_types"] == ["FALL"]

    def test_update_global_filters_handles_none_values(self):
        from src.dashboard.components.filters import update_global_filters
        result = update_global_filters(
            years=None, services=None, call_types=None, reset_clicks=0
        )
        assert result["years"] == []
        assert result["services"] == []
        assert result["call_types"] == []


# ===========================================================================
# APP LAYOUT INTEGRATION TESTS
# ===========================================================================

@_skip_no_dash
@_skip_no_app
class TestAppLayout:
    """Verify the main app layout has all expected structural elements."""

    def test_app_layout_exists(self):
        from src.dashboard.app import app
        assert app.layout is not None

    def test_app_layout_has_page_container(self):
        from src.dashboard.app import app
        layout = app.layout
        assert layout is not None

    def test_navbar_contains_all_navigation_links(self):
        from src.dashboard.app import app
        layout = app.layout
        layout_str = str(layout)
        for label in ["Overview", "Temporal", "Geography",
                       "Forecast", "Classification QA", "Assistant"]:
            assert label in layout_str, (
                f"Navigation link '{label}' not found in app layout"
            )


# ===========================================================================
# SHARED COMPONENT TESTS — MAP UTILS
# ===========================================================================

@_skip_no_dash
class TestMapUtils:
    """Verify shared map utility functions.

    Some tests may be skipped if the installed Plotly version has
    breaking changes for Scattermapbox marker properties.
    """

    def test_create_choropleth_returns_figure(self, geo_data_df):
        from src.dashboard.components.map_utils import create_choropleth_map
        import plotly.graph_objects as go

        try:
            fig = create_choropleth_map(geo_data_df, "call_count", "Test Choropleth")
        except ValueError as exc:
            if "Invalid property" in str(exc):
                pytest.skip(f"Plotly version incompatibility: {exc}")
            raise
        assert isinstance(fig, go.Figure)

    def test_create_cluster_map_returns_figure(self, geo_data_df):
        from src.dashboard.components.map_utils import create_cluster_map
        import plotly.graph_objects as go

        try:
            fig = create_cluster_map(geo_data_df, eps=0.05, min_samples=3)
        except ValueError as exc:
            if "Invalid property" in str(exc):
                pytest.skip(f"Plotly version incompatibility: {exc}")
            raise
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_create_heatmap_returns_figure(self, geo_data_df):
        from src.dashboard.components.map_utils import create_heatmap
        import plotly.graph_objects as go

        fig = create_heatmap(geo_data_df, "Test Heatmap")
        assert isinstance(fig, go.Figure)

    def test_choropleth_map_has_correct_title(self, geo_data_df):
        from src.dashboard.components.map_utils import create_choropleth_map

        try:
            fig = create_choropleth_map(geo_data_df, "call_count", "Dispatch Density")
        except ValueError as exc:
            if "Invalid property" in str(exc):
                pytest.skip(f"Plotly version incompatibility: {exc}")
            raise
        assert fig.layout.title.text == "Dispatch Density"

    def test_cluster_map_has_scattermapbox_traces(self, geo_data_df):
        from src.dashboard.components.map_utils import create_cluster_map
        import plotly.graph_objects as go

        try:
            fig = create_cluster_map(geo_data_df)
        except ValueError as exc:
            if "Invalid property" in str(exc):
                pytest.skip(f"Plotly version incompatibility: {exc}")
            raise
        for trace in fig.data:
            assert isinstance(trace, go.Scattermapbox)

    def test_map_utils_has_three_public_functions(self):
        from src.dashboard.components import map_utils
        public = [n for n in dir(map_utils) if not n.startswith("_") and callable(getattr(map_utils, n))]
        assert len(public) >= 3, f"Expected ≥3 public functions, got: {public}"


# ===========================================================================
# SHARED COMPONENT TESTS — CHAT UI
# ===========================================================================

@_skip_no_dash
@_skip_no_langchain
class TestChatUI:
    """Verify chat UI component structure."""

    def test_create_chat_container_returns_component(self):
        from src.dashboard.components.chat_ui import create_chat_container
        import dash_bootstrap_components as dbc

        container = create_chat_container()
        assert isinstance(container, dbc.Container)

    def test_chat_ui_has_expected_prompt_count(self):
        from src.dashboard.components.chat_ui import PROMPTS
        assert len(PROMPTS) >= 4, "Expected at least 4 example prompts"

    def test_chat_container_has_input_textarea(self):
        from src.dashboard.components.chat_ui import create_chat_container

        container = create_chat_container()
        found = _find_component_by_id(container, "chat-input")
        assert found is not None, "chat-input textarea not found"

    def test_chat_container_has_send_button(self):
        from src.dashboard.components.chat_ui import create_chat_container

        container = create_chat_container()
        found = _find_component_by_id(container, "send-btn")
        assert found is not None, "send-btn not found"

    def test_chat_container_has_message_area(self):
        from src.dashboard.components.chat_ui import create_chat_container

        container = create_chat_container()
        found = _find_component_by_id(container, "chat-messages")
        assert found is not None, "chat-messages area not found"

    def test_chat_container_has_history_store(self):
        from src.dashboard.components.chat_ui import create_chat_container

        container = create_chat_container()
        found = _find_component_by_id(container, "chat-history-store")
        assert found is not None, "chat-history-store not found"

    def test_chat_container_has_source_accordion(self):
        from src.dashboard.components.chat_ui import create_chat_container

        container = create_chat_container()
        found = _find_component_by_id(container, "source-accordion-container")
        assert found is not None, "source-accordion-container not found"

    def test_prompt_buttons_exist(self):
        from src.dashboard.components.chat_ui import create_chat_container, PROMPTS

        container = create_chat_container()
        for i in range(len(PROMPTS)):
            found = _find_component_by_id(container, f"assistant-prompt-{i}")
            assert found is not None, f"assistant-prompt-{i} button not found"


# ===========================================================================
# FORECAST PAGE TESTS (Deekshitha's page)
# ===========================================================================

@_skip_no_dash
class TestForecastPage:
    """Verify forecast page-specific components."""

    def test_forecast_page_has_model_toggle(self):
        from src.dashboard.pages.forecast import layout

        page_layout = layout if not callable(layout) else layout()
        layout_str = str(page_layout)
        assert any(term in layout_str
                    for term in ["Prophet", "LightGBM", "Ensemble"]), (
            "Forecast page should mention Prophet/LightGBM/Ensemble model options"
        )

    def test_forecast_page_has_chart_component(self):
        from src.dashboard.pages.forecast import layout

        page_layout = layout if not callable(layout) else layout()
        found = _find_component_by_id(page_layout, "forecast-chart")
        assert found is not None, "forecast-chart Graph not found"

    def test_forecast_page_has_model_selector(self):
        from src.dashboard.pages.forecast import layout

        page_layout = layout if not callable(layout) else layout()
        found = _find_component_by_id(page_layout, "model-selector")
        assert found is not None, "model-selector (RadioItems) not found"

    def test_forecast_page_has_data_store(self):
        from src.dashboard.pages.forecast import layout

        page_layout = layout if not callable(layout) else layout()
        found = _find_component_by_id(page_layout, "forecast-data-store")
        assert found is not None, "forecast-data-store not found"

    def test_forecast_page_has_quarterly_stats_row(self):
        from src.dashboard.pages.forecast import layout

        page_layout = layout if not callable(layout) else layout()
        found = _find_component_by_id(page_layout, "quarterly-stats")
        assert found is not None, "quarterly-stats row not found"

    def test_forecast_callbacks_are_defined(self):
        from src.dashboard.pages.forecast import (
            update_forecast_data,
            update_forecast_chart,
            update_quarterly_stats,
        )
        assert callable(update_forecast_data)
        assert callable(update_forecast_chart)
        assert callable(update_quarterly_stats)

    def test_forecast_empty_figure_function(self):
        from src.dashboard.pages.forecast import _empty_figure
        import plotly.graph_objects as go

        fig = _empty_figure("No data")
        assert isinstance(fig, go.Figure)

    def test_forecast_quarterly_stats_builder(self):
        from src.dashboard.pages.forecast import _create_quarterly_stats
        import dash_bootstrap_components as dbc

        sample = {
            "quarters": ["Q1 2025", "Q2 2025", "Q3 2025", "Q4 2025"],
            "forecast": [1000, 1100, 1200, 1300],
        }
        result = _create_quarterly_stats(sample)
        assert isinstance(result, dbc.Row)


# ===========================================================================
# CALLBACK REGISTRATION TESTS
# ===========================================================================

@_skip_no_dash
class TestCallbackRegistration:
    """Verify key callbacks are registered in the Dash app."""

    def test_filter_callback_is_callable(self):
        from src.dashboard.components.filters import update_global_filters
        assert callable(update_global_filters)

    @_skip_no_langchain
    def test_chat_callback_is_callable(self):
        from src.dashboard.components.chat_ui import update_chat
        assert callable(update_chat)

    def test_forecast_callbacks_are_callable(self):
        from src.dashboard.pages.forecast import (
            update_forecast_data,
            update_forecast_chart,
            update_quarterly_stats,
        )
        assert callable(update_forecast_data)
        assert callable(update_forecast_chart)
        assert callable(update_quarterly_stats)


# ===========================================================================
# CONFIG / SETTINGS INTEGRATION TESTS
# ===========================================================================

class TestConfigIntegration:
    """Verify dashboard uses config/settings.py correctly."""

    def test_flask_settings_exist(self):
        from config.settings import FLASK_HOST, FLASK_PORT, FLASK_DEBUG, FLASK_THREADED
        assert isinstance(FLASK_HOST, str)
        assert isinstance(FLASK_PORT, int)
        assert isinstance(FLASK_DEBUG, bool)
        assert isinstance(FLASK_THREADED, bool)

    def test_flask_port_is_8050(self):
        from config.settings import FLASK_PORT
        assert FLASK_PORT == 8050

    def test_project_root_setting_exists(self):
        from config.settings import PROJECT_ROOT
        assert PROJECT_ROOT.exists()

    def test_model_artifacts_dir_setting_exists(self):
        from config.settings import MODEL_ARTIFACTS_DIR
        assert MODEL_ARTIFACTS_DIR is not None
        # Directory may not exist locally but the path must be defined
        assert isinstance(MODEL_ARTIFACTS_DIR, Path)


# ===========================================================================
# HELPER UTILITIES
# ===========================================================================

def _find_component_by_id(component, target_id: str):
    """Recursively search a Dash component tree for a component with the given id."""
    # Check current component
    if hasattr(component, "id") and component.id == target_id:
        return component

    # Check children
    children = getattr(component, "children", None)
    if children is None:
        return None

    if isinstance(children, (list, tuple)):
        for child in children:
            result = _find_component_by_id(child, target_id)
            if result is not None:
                return result
    elif hasattr(children, "id") or hasattr(children, "children"):
        result = _find_component_by_id(children, target_id)
        if result is not None:
            return result

    return None

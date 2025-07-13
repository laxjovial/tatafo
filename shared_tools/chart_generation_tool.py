# shared_tools/chart_generation_tool.py

import logging
import json
import uuid
import os
from typing import Dict, Any, List, Optional

# Import plotting libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np # For histogram bins, etc.

# Try importing seaborn and plotly, but handle cases where they might not be installed
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    sns = None
    SEABORN_AVAILABLE = False
    logging.warning("Seaborn not installed. Some chart types may not be available.")

try:
    import plotly.graph_objects as go
    from plotly.offline import plot as plotly_plot_html # For saving HTML
    PLOTLY_AVAILABLE = True
except ImportError:
    go = None
    plotly_plot_html = None
    PLOTLY_AVAILABLE = False
    logging.warning("Plotly not installed. Some chart types may not be available.")


from langchain_core.tools import tool

# Import config_manager and user_manager for RBAC checks
from config.config_manager import config_manager
from utils.user_manager import get_user_tier_capability

# Import UserProfile for type hinting
from backend.models.user_models import UserProfile

logger = logging.getLogger(__name__)

# Base directory for storing generated charts
BASE_CHART_DIR = Path("charts")

class ChartTools:
    """
    A collection of tools for generating various types of charts and visualizations
    from data, with granular RBAC control over chart types and underlying libraries.
    """
    def __init__(self, config_manager):
        self.config_manager = config_manager
        logger.info("ChartTools initialized.")

    @tool
    async def generate_and_save_chart(
        self,
        data_json: str,
        chart_type: str,
        x_column: Optional[str] = None, # Optional for pie, histogram, boxplot
        y_column: Optional[str] = None, # Optional for pie, histogram, boxplot
        color_column: Optional[str] = None, # For seaborn/plotly hue/color
        title: str = "Generated Chart",
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        user_context: UserProfile = None, # UserProfile for RBAC
        library: str = "matplotlib", # Preferred plotting library: "matplotlib", "seaborn", "plotly"
        export_format: str = "png", # "png", "jpeg", "svg", "html" (for plotly)
        **kwargs: Any # For additional chart-specific parameters (e.g., bins for histogram, orientation for bar)
    ) -> str:
        """
        Generates a chart (e.g., line, bar, scatter, histogram, boxplot, pie, area) from JSON data
        and saves it as an image or HTML file. Supports Matplotlib, Seaborn, and Plotly.
        The file path to the generated chart is returned.

        Args:
            data_json (str): A JSON string representing the data to plot.
                             Expected format: a list of dictionaries, where each dictionary
                             is a row and keys are column names.
                             Example: '[{"date": "2023-01-01", "value": 10}, {"date": "2023-01-02", "value": 12}]'
            chart_type (str): The type of chart to generate. Supported:
                              'line', 'bar', 'scatter' (Matplotlib, Seaborn, Plotly)
                              'histogram', 'boxplot' (Seaborn, Plotly)
                              'pie', 'area' (Matplotlib, Plotly)
            x_column (str, optional): The name of the column for the X-axis. Required for most plots, optional for pie/histogram/boxplot.
            y_column (str, optional): The name of the column for the Y-axis. Required for most plots, optional for pie/histogram/boxplot.
            color_column (str, optional): Column to use for coloring/grouping (e.g., 'hue' in Seaborn).
            title (str, optional): The title of the chart. Defaults to "Generated Chart".
            x_label (str, optional): Label for the X-axis. Defaults to x_column.
            y_label (str, optional): Label for the Y-axis. Defaults to y_column.
            user_context (UserProfile): The user's profile for RBAC checks.
            library (str, optional): The preferred plotting library ('matplotlib', 'seaborn', 'plotly'). Defaults to 'matplotlib'.
            export_format (str, optional): The format to save the chart ('png', 'jpeg', 'svg', 'html'). Defaults to 'png'.
                                           'html' is only supported for Plotly charts.
            **kwargs: Additional chart-specific parameters (e.g., 'bins' for histogram, 'orientation' for bar, 'names_column' and 'values_column' for pie).

        Returns:
            str: The file path to the generated chart file, or an error message.
        """
        if user_context is None:
            user_context = UserProfile(user_id="default", username="CLI_User", email="cli@example.com", tier="free", roles=["user"])

        logger.info(f"Tool: generate_and_save_chart called for type '{chart_type}' using library '{library}' for user: {user_context.user_id}")

        # RBAC Check: General chart generation access
        if not get_user_tier_capability(user_context.user_id, 'chart_generation_enabled', False, user_tier=user_context.tier, user_roles=user_context.roles):
            return "Error: Chart generation is not enabled for your current tier. Please upgrade your plan."
        
        # RBAC Check: Specific library access
        allowed_libraries = get_user_tier_capability(user_context.user_id, 'chart_library_access', {'matplotlib': True}, user_tier=user_context.tier, user_roles=user_context.roles)
        if not allowed_libraries.get(library, False):
            return f"Error: Access to the '{library}' plotting library is not enabled for your current tier."

        # RBAC Check: Specific chart type access
        allowed_chart_types = get_user_tier_capability(user_context.user_id, 'chart_type_access', {'line': True, 'bar': True, 'scatter': True}, user_tier=user_context.tier, user_roles=user_context.roles)
        if not allowed_chart_types.get(chart_type, False):
            return f"Error: The chart type '{chart_type}' is not enabled for your current tier."
        
        # RBAC Check: Export format
        is_export_allowed = get_user_tier_capability(user_context.user_id, 'chart_export_enabled', False, user_tier=user_context.tier, user_roles=user_context.roles)
        if export_format != "png" and not is_export_allowed: # PNG is generally always allowed as default
            return f"Error: Exporting charts to '{export_format}' format is not enabled for your current tier."
        if export_format == "html" and library != "plotly":
            return "Error: HTML export is only supported for Plotly charts."
        if export_format == "html" and not PLOTLY_AVAILABLE:
            return "Error: Plotly library is not available for HTML export."


        try:
            data = json.loads(data_json)
            if not isinstance(data, list) or not all(isinstance(d, dict) for d in data):
                raise ValueError("Input data_json must be a JSON array of objects.")
            
            if not data:
                return "Error: No data provided to generate chart."

            df = pd.DataFrame(data)

            # Validate columns based on chart type requirements
            if chart_type in ['line', 'bar', 'scatter', 'area']:
                if not x_column or not y_column:
                    return f"Error: '{chart_type}' chart type requires both 'x_column' and 'y_column'."
                if x_column not in df.columns or y_column not in df.columns:
                    return f"Error: Specified columns '{x_column}' or '{y_column}' not found in data."
            elif chart_type in ['histogram', 'boxplot']:
                if not x_column and not y_column: # One of them should be present
                    return f"Error: '{chart_type}' chart type requires at least 'x_column' or 'y_column'."
                if x_column and x_column not in df.columns:
                    return f"Error: Specified x_column '{x_column}' not found in data."
                if y_column and y_column not in df.columns:
                    return f"Error: Specified y_column '{y_column}' not found in data."
            elif chart_type == 'pie':
                names_column = kwargs.get('names_column')
                values_column = kwargs.get('values_column')
                if not names_column or not values_column:
                    return "Error: 'pie' chart type requires 'names_column' and 'values_column' in kwargs."
                if names_column not in df.columns or values_column not in df.columns:
                    return f"Error: Specified columns '{names_column}' or '{values_column}' not found in data for pie chart."

            # Create user-specific chart directory
            user_chart_dir = BASE_CHART_DIR / user_context.user_id
            user_chart_dir.mkdir(parents=True, exist_ok=True)

            # Generate unique filename based on export format
            filename = f"chart_{uuid.uuid4().hex}.{export_format}"
            file_path = user_chart_dir / filename

            # --- Plotting Logic based on Library and Chart Type ---
            if library == "matplotlib":
                plt.figure(figsize=(10, 6))
                if chart_type == 'line':
                    plt.plot(df[x_column], df[y_column])
                elif chart_type == 'bar':
                    plt.bar(df[x_column], df[y_column])
                elif chart_type == 'scatter':
                    plt.scatter(df[x_column], df[y_column])
                elif chart_type == 'histogram':
                    # Use x_column if provided, else y_column
                    data_for_hist = df[x_column] if x_column else df[y_column]
                    plt.hist(data_for_hist.dropna(), bins=kwargs.get('bins', 10))
                elif chart_type == 'boxplot':
                    # Use x_column if provided, else y_column
                    data_for_box = df[x_column] if x_column else df[y_column]
                    plt.boxplot(data_for_box.dropna())
                elif chart_type == 'pie':
                    plt.pie(df[values_column], labels=df[names_column], autopct='%1.1f%%', startangle=90)
                    plt.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
                elif chart_type == 'area':
                    plt.fill_between(df[x_column], df[y_column])
                else:
                    plt.close()
                    return f"Error: Matplotlib does not support chart type '{chart_type}' or it's not implemented."

                plt.title(title)
                plt.xlabel(x_label if x_label else x_column)
                plt.ylabel(y_label if y_label else y_column)
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(file_path)
                plt.close()

            elif library == "seaborn":
                if not SEABORN_AVAILABLE:
                    return "Error: Seaborn library is not installed or available."
                
                plt.figure(figsize=(10, 6))
                
                if chart_type == 'line':
                    sns.lineplot(x=x_column, y=y_column, data=df, hue=color_column)
                elif chart_type == 'bar':
                    sns.barplot(x=x_column, y=y_column, data=df, hue=color_column)
                elif chart_type == 'scatter':
                    sns.scatterplot(x=x_column, y=y_column, data=df, hue=color_column)
                elif chart_type == 'histogram':
                    sns.histplot(data=df, x=x_column, y=y_column, hue=color_column, bins=kwargs.get('bins', 10))
                elif chart_type == 'boxplot':
                    sns.boxplot(x=x_column, y=y_column, data=df, hue=color_column)
                elif chart_type == 'area':
                    # Seaborn doesn't have a direct 'area' plot like fill_between, often done with lineplot + fill
                    # For simplicity, we'll use matplotlib's fill_between if area is requested with seaborn
                    # or indicate it's not directly supported.
                    plt.close()
                    return "Error: Seaborn does not have a direct 'area' chart type. Consider using Matplotlib or Plotly for area charts."
                elif chart_type == 'pie':
                    plt.close()
                    return "Error: Seaborn does not directly support 'pie' charts. Consider using Matplotlib or Plotly."
                else:
                    plt.close()
                    return f"Error: Seaborn does not support chart type '{chart_type}' or it's not implemented."

                plt.title(title)
                plt.xlabel(x_label if x_label else x_column)
                plt.ylabel(y_label if y_label else y_column)
                plt.tight_layout()
                plt.savefig(file_path)
                plt.close()

            elif library == "plotly":
                if not PLOTLY_AVAILABLE:
                    return "Error: Plotly library is not installed or available."
                
                fig = None
                if chart_type == 'line':
                    fig = go.Figure(data=[go.Scatter(x=df[x_column], y=df[y_column], mode='lines', name=y_column)])
                elif chart_type == 'bar':
                    fig = go.Figure(data=[go.Bar(x=df[x_column], y=df[y_column], name=y_column)])
                elif chart_type == 'scatter':
                    fig = go.Figure(data=[go.Scatter(x=df[x_column], y=df[y_column], mode='markers', name=y_column)])
                elif chart_type == 'histogram':
                    fig = go.Figure(data=[go.Histogram(x=df[x_column] if x_column else df[y_column], nbinsx=kwargs.get('bins', 10))])
                elif chart_type == 'boxplot':
                    fig = go.Figure(data=[go.Box(y=df[y_column] if y_column else df[x_column])])
                elif chart_type == 'pie':
                    fig = go.Figure(data=[go.Pie(labels=df[names_column], values=df[values_column])])
                elif chart_type == 'area':
                    fig = go.Figure(data=[go.Scatter(x=df[x_column], y=df[y_column], fill='tozeroy', mode='lines', name=y_column)])
                else:
                    return f"Error: Plotly does not support chart type '{chart_type}' or it's not implemented."

                if fig:
                    fig.update_layout(title_text=title, xaxis_title=x_label if x_label else x_column, yaxis_title=y_label if y_label else y_column)
                    if export_format == "html":
                        plotly_plot_html(fig, filename=str(file_path), auto_open=False)
                    else:
                        fig.write_image(str(file_path))
                else:
                    return f"Error: Failed to create Plotly figure for chart type '{chart_type}'."

            else:
                return f"Error: Unsupported plotting library '{library}'. Supported: 'matplotlib', 'seaborn', 'plotly'."

            logger.info(f"Chart '{chart_type}' using '{library}' saved to: {file_path}")
            return str(file_path)
        
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON data provided: {data_json}", exc_info=True)
            return "Error: Invalid JSON data provided for chart generation."
        except ValueError as ve:
            logger.error(f"Data or plotting error for chart generation: {ve}", exc_info=True)
            return f"Error: Data or plotting failed for chart generation: {ve}"
        except Exception as e:
            logger.error(f"An unexpected error occurred during chart generation: {e}", exc_info=True)
            return f"An unexpected error occurred during chart generation: {e}"

# CLI Test (optional)
if __name__ == "__main__":
    import shutil
    from unittest.mock import MagicMock, patch
    import sys

    logging.basicConfig(level=logging.INFO)

    # Mock UserProfile for testing
    from backend.models.user_models import UserProfile
    mock_user_pro_profile = UserProfile(user_id="mock_pro_token", username="ProUser", email="pro@example.com", tier="pro", roles=["user"])
    mock_user_premium_profile = UserProfile(user_id="mock_premium_token", username="PremiumUser", email="premium@example.com", tier="premium", roles=["user"])
    mock_user_free_profile = UserProfile(user_id="mock_free_token", username="FreeUser", email="free@example.com", tier="free", roles=["user"])
    mock_user_admin_profile = UserProfile(user_id="mock_admin_token", username="AdminUser", email="admin@example.com", tier="admin", roles=["user", "admin"])


    # Mock Streamlit secrets and config_manager for local testing
    class MockSecrets:
        def __init__(self):
            self.openai = {"api_key": "sk-mock-openai-key-12345"}
            self.google = {"api_key": "AIzaSy-mock-google-key"}
            self.user_tokens = {
                "free_user_token": "mock_free_token",
                "pro_user_token": "mock_pro_token",
                "premium_user_token": "mock_premium_token",
                "admin_user_token": "mock_admin_token"
            }
            self.firebase_config = "{}" # Mock empty config for Firebase if not set

        def get(self, key, default=None):
            parts = key.split('.')
            val = self
            for part in parts:
                if hasattr(val, part):
                    val = getattr(val, part)
                elif isinstance(val, dict) and part in val:
                    val = val[part]
                else:
                    return default
            return val
    
    class MockConfigManager:
        _instance = None
        _is_loaded = False
        def __init__(self):
            if MockConfigManager._instance is not None:
                raise Exception("ConfigManager is a singleton. Use get_instance().")
            MockConfigManager._instance = self
            self._config_data = {
                'llm': {'max_summary_input_chars': 10000},
                'rag': {'chunk_size': 500, 'chunk_overlap': 50, 'max_query_results_k': 10},
                'web_scraping': {'user_agent': 'Mozilla/5.0 (Test; Python)', 'timeout_seconds': 5, 'max_search_results': 5},
                'tiers': {}, # This will be overridden by tiers.yaml
                'default_user_tier': 'free',
                'default_user_roles': ['user']
            }
            self._is_loaded = True
        
        def get(self, key, default=None):
            parts = key.split('.')
            val = self._config_data
            for part in parts:
                if isinstance(val, dict) and part in val:
                    val = val[part]
                else:
                    return default
            return val
        
        def get_secret(self, key, default=None):
            return st.secrets.get(key, default)

    # Mock user_manager.get_user_tier_capability for testing RBAC
    class MockUserManager:
        _mock_users = {
            "mock_free_token": {"user_id": "mock_free_token", "username": "FreeUser", "email": "free@example.com", "tier": "free", "roles": ["user"]},
            "mock_pro_token": {"user_id": "mock_pro_token", "username": "ProUser", "email": "pro@example.com", "tier": "pro", "roles": ["user"]},
            "mock_premium_token": {"user_id": "mock_premium_token", "username": "PremiumUser", "email": "premium@example.com", "tier": "premium", "roles": ["user"]},
            "mock_admin_token": {"user_id": "mock_admin_token", "username": "AdminUser", "email": "admin@example.com", "tier": "admin", "roles": ["user", "admin"]},
        }
        _rbac_capabilities = {
            'capabilities': {
                'chart_generation_enabled': {
                    'default': False,
                    'roles': {'pro': True, 'premium': True, 'admin': True}
                },
                'chart_export_enabled': {
                    'default': False,
                    'roles': {'premium': True, 'admin': True}
                },
                'chart_library_access': {
                    'default': {'matplotlib': True},
                    'tiers': {
                        'pro': {'matplotlib': True, 'seaborn': False, 'plotly': False},
                        'premium': {'matplotlib': True, 'seaborn': True, 'plotly': True},
                        'admin': {'matplotlib': True, 'seaborn': True, 'plotly': True}
                    }
                },
                'chart_type_access': {
                    'default': {'line': True, 'bar': True, 'scatter': True},
                    'tiers': {
                        'pro': {'line': True, 'bar': True, 'scatter': True, 'histogram': True, 'boxplot': True},
                        'premium': {'line': True, 'bar': True, 'scatter': True, 'histogram': True, 'boxplot': True, 'pie': True, 'area': True},
                        'admin': {'line': True, 'bar': True, 'scatter': True, 'histogram': True, 'boxplot': True, 'pie': True, 'area': True}
                    }
                }
            }
        }
        _tier_hierarchy = {
            "free": 0, "user": 1, "basic": 2, "pro": 3, "premium": 4, "admin": 99
        }

        def get_current_user(self) -> Dict[str, Any]:
            return getattr(self, '_current_mock_user', {})

        def get_user_tier_capability(self, user_token: Optional[str], capability_key: str, default_value: Any = None, user_tier: Optional[str] = None, user_roles: Optional[List[str]] = None) -> Any:
            # Use provided user_tier and user_roles if available, otherwise lookup from mock_users
            if user_tier is None or user_roles is None:
                user_info = self._mock_users.get(user_token, {})
                user_tier = user_info.get('tier', 'free')
                user_roles = user_info.get('roles', [])

            if "admin" in user_roles:
                # Admin has full access to all chart types and libraries
                if capability_key == 'chart_library_access':
                    return {'matplotlib': True, 'seaborn': True, 'plotly': True}
                if capability_key == 'chart_type_access':
                    return {'line': True, 'bar': True, 'scatter': True, 'histogram': True, 'boxplot': True, 'pie': True, 'area': True}
                if isinstance(default_value, bool): return True
                if isinstance(default_value, (int, float)): return float('inf')
                return default_value # For other types, return default (e.g., string values)
            
            capability_config = self._rbac_capabilities.get('capabilities', {}).get(capability_key)
            if not capability_config:
                return default_value

            for role in user_roles:
                if role in capability_config.get('roles', {}):
                    return capability_config['roles'][role]
            
            if user_tier in capability_config.get('tiers', {}):
                return capability_config['tiers'][user_tier]

            return capability_config.get('default', default_value)

    # Patch the actual imports for testing
    import streamlit as st_mock
    if not hasattr(st_mock, 'secrets'):
        st_mock.secrets = MockSecrets()
    
    # Mock config_manager and user_manager
    sys.modules['config.config_manager'].config_manager = MockConfigManager()
    sys.modules['config.config_manager'].ConfigManager = MockConfigManager
    sys.modules['utils.user_manager'].get_user_tier_capability = MockUserManager().get_user_tier_capability
    # Also set the internal _RBAC_CAPABILITIES and _TIER_HIERARCHY for direct access in other modules if they use it
    sys.modules['utils.user_manager']._RBAC_CAPABILITIES = MockUserManager()._rbac_capabilities
    sys.modules['utils.user_manager']._TIER_HIERARCHY = MockUserManager()._tier_hierarchy


    print("\n--- Testing ChartTools functions ---")
    
    # Instantiate ChartTools for testing
    chart_tools_instance = ChartTools(config_manager=sys.modules['config.config_manager'].config_manager)

    sample_data = [
        {"month": "Jan", "sales": 100, "expenses": 50, "region": "East"},
        {"month": "Feb", "sales": 120, "expenses": 60, "region": "West"},
        {"month": "Mar", "sales": 90, "expenses": 55, "region": "East"},
        {"month": "Apr", "sales": 130, "expenses": 70, "region": "West"},
        {"month": "May", "sales": 110, "expenses": 65, "region": "East"},
        {"month": "Jun", "sales": 140, "expenses": 75, "region": "West"},
    ]
    sample_data_json = json.dumps(sample_data)

    # Data for histogram/boxplot
    numerical_data = [{"value": x} for x in np.random.normal(loc=50, scale=10, size=100).tolist()]
    numerical_data_json = json.dumps(numerical_data)

    # Data for pie chart
    pie_data = [
        {"category": "A", "count": 30},
        {"category": "B", "count": 20},
        {"category": "C", "count": 50},
    ]
    pie_data_json = json.dumps(pie_data)


    # Clean up charts directory from previous runs
    if BASE_CHART_DIR.exists():
        shutil.rmtree(BASE_CHART_DIR)
    BASE_CHART_DIR.mkdir(exist_ok=True)


    # Test 1: Pro user - Matplotlib Line chart (allowed)
    print("\n--- Test 1: Pro user, Matplotlib Line chart ---")
    chart_path_pro_line = await chart_tools_instance.generate_and_save_chart(
        data_json=sample_data_json,
        chart_type="line",
        x_column="month",
        y_column="sales",
        title="Monthly Sales (Pro User)",
        user_context=mock_user_pro_profile,
        library="matplotlib"
    )
    print(f"Chart path (Pro user, Matplotlib Line): {chart_path_pro_line}")
    assert isinstance(chart_path_pro_line, str) and Path(chart_path_pro_line).exists()
    assert Path(chart_path_pro_line).parent.name == mock_user_pro_profile.user_id
    print("Test 1 Passed: Matplotlib Line chart generated for Pro user.")

    # Test 2: Pro user - Seaborn Bar chart (denied by RBAC)
    print("\n--- Test 2: Pro user, Seaborn Bar chart (Denied) ---")
    error_message_pro_seaborn = await chart_tools_instance.generate_and_save_chart(
        data_json=sample_data_json,
        chart_type="bar",
        x_column="month",
        y_column="sales",
        user_context=mock_user_pro_profile,
        library="seaborn" # Pro user not allowed Seaborn
    )
    print(f"Result (Pro user, Seaborn): {error_message_pro_seaborn}")
    assert "Error: Access to the 'seaborn' plotting library is not enabled for your current tier." in error_message_pro_seaborn
    print("Test 2 Passed: Seaborn access correctly denied for Pro user.")

    # Test 3: Premium user - Seaborn Histogram (allowed)
    print("\n--- Test 3: Premium user, Seaborn Histogram ---")
    chart_path_premium_hist = await chart_tools_instance.generate_and_save_chart(
        data_json=numerical_data_json,
        chart_type="histogram",
        x_column="value",
        title="Value Distribution (Premium User, Seaborn)",
        user_context=mock_user_premium_profile,
        library="seaborn"
    )
    print(f"Chart path (Premium user, Seaborn Hist): {chart_path_premium_hist}")
    assert isinstance(chart_path_premium_hist, str) and Path(chart_path_premium_hist).exists()
    print("Test 3 Passed: Seaborn Histogram generated for Premium user.")

    # Test 4: Premium user - Plotly Pie chart (allowed)
    print("\n--- Test 4: Premium user, Plotly Pie chart ---")
    chart_path_premium_pie = await chart_tools_instance.generate_and_save_chart(
        data_json=pie_data_json,
        chart_type="pie",
        names_column="category",
        values_column="count",
        title="Category Distribution (Premium User, Plotly)",
        user_context=mock_user_premium_profile,
        library="plotly",
        export_format="html" # Test HTML export
    )
    print(f"Chart path (Premium user, Plotly Pie HTML): {chart_path_premium_pie}")
    assert isinstance(chart_path_premium_pie, str) and Path(chart_path_premium_pie).exists()
    assert chart_path_premium_pie.endswith(".html")
    print("Test 4 Passed: Plotly Pie chart (HTML) generated for Premium user.")

    # Test 5: Free user - Chart generation denied (general RBAC)
    print("\n--- Test 5: Free user, Chart generation denied (general RBAC) ---")
    error_message_free_general = await chart_tools_instance.generate_and_save_chart(
        data_json=sample_data_json,
        chart_type="line",
        x_column="month",
        y_column="sales",
        user_context=mock_user_free_profile,
        library="matplotlib"
    )
    print(f"Result (Free user, general denial): {error_message_free_general}")
    assert "Error: Chart generation is not enabled for your current tier." in error_message_free_general
    print("Test 5 Passed: General chart generation correctly denied for Free user.")

    # Test 6: Admin user - Plotly Area chart (allowed, admin override)
    print("\n--- Test 6: Admin user, Plotly Area chart ---")
    chart_path_admin_area = await chart_tools_instance.generate_and_save_chart(
        data_json=sample_data_json,
        chart_type="area",
        x_column="month",
        y_column="sales",
        title="Monthly Sales Area (Admin User, Plotly)",
        user_context=mock_user_admin_profile,
        library="plotly"
    )
    print(f"Chart path (Admin user, Plotly Area): {chart_path_admin_area}")
    assert isinstance(chart_path_admin_area, str) and Path(chart_path_admin_area).exists()
    print("Test 6 Passed: Plotly Area chart generated for Admin user.")

    # Test 7: Pro user - Pie chart (type denied by RBAC)
    print("\n--- Test 7: Pro user, Pie chart (type denied) ---")
    error_message_pro_pie = await chart_tools_instance.generate_and_save_chart(
        data_json=pie_data_json,
        chart_type="pie",
        names_column="category",
        values_column="count",
        user_context=mock_user_pro_profile,
        library="matplotlib" # Matplotlib is allowed, but pie chart type is not for Pro
    )
    print(f"Result (Pro user, Pie type denied): {error_message_pro_pie}")
    assert "Error: The chart type 'pie' is not enabled for your current tier." in error_message_pro_pie
    print("Test 7 Passed: Pie chart type correctly denied for Pro user.")

    # Test 8: Pro user - HTML export (denied by RBAC)
    print("\n--- Test 8: Pro user, HTML export (denied) ---")
    error_message_pro_html_export = await chart_tools_instance.generate_and_save_chart(
        data_json=sample_data_json,
        chart_type="line",
        x_column="month",
        y_column="sales",
        user_context=mock_user_pro_profile,
        library="matplotlib",
        export_format="html" # HTML export not allowed for Pro
    )
    print(f"Result (Pro user, HTML export denied): {error_message_pro_html_export}")
    assert "Error: Exporting charts to 'html' format is not enabled for your current tier." in error_message_pro_html_export
    print("Test 8 Passed: HTML export correctly denied for Pro user.")


    # Test 9: Invalid JSON data
    print("\n--- Test 9: Invalid JSON data ---")
    invalid_json_result = await chart_tools_instance.generate_and_save_chart(
        data_json="not a json string",
        chart_type="line",
        x_column="month",
        y_column="sales",
        user_context=mock_user_pro_profile,
        library="matplotlib"
    )
    print(f"Result (Invalid JSON): {invalid_json_result}")
    assert "Error: Invalid JSON data provided for chart generation." in invalid_json_result
    print("Test 9 Passed: Invalid JSON data handled.")

    # Test 10: Missing columns for required chart type
    print("\n--- Test 10: Missing columns for required chart type ---")
    missing_columns_result = await chart_tools_instance.generate_and_save_chart(
        data_json=sample_data_json,
        chart_type="line",
        x_column="non_existent_col_x",
        y_column="non_existent_col_y",
        user_context=mock_user_pro_profile,
        library="matplotlib"
    )
    print(f"Result (Missing columns): {missing_columns_result}")
    assert "Error: Specified columns 'non_existent_col_x' or 'non_existent_col_y' not found in data." in missing_columns_result
    print("Test 10 Passed: Missing columns handled.")

    print("\nAll ChartTools tests completed.")

    # Clean up generated chart files and directories
    if BASE_CHART_DIR.exists():
        shutil.rmtree(BASE_CHART_DIR)
        print(f"\nCleaned up chart directory: {BASE_CHART_DIR}")

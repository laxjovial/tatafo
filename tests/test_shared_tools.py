import pytest
from unittest.mock import patch, AsyncMock
from shared_tools.chart_generation_tool import ChartTools
from shared_tools.doc_summarizer import summarize_document
from shared_tools.export_utils import export_data_to_csv
from shared_tools.python_interpreter_tool import PythonInterpreterTool
from shared_tools.scrapper_tool import scrape_web
from backend.models.user_models import UserProfile

@pytest.fixture
def user_profile():
    return UserProfile(user_id="test_user", tier="premium", roles=["user"])

@pytest.mark.asyncio
@patch('shared_tools.chart_generation_tool.plt.savefig')
async def test_generate_and_save_chart(mock_savefig, user_profile):
    chart_tools = ChartTools(config_manager=None)
    data_json = '[{"month": "Jan", "sales": 100}, {"month": "Feb", "sales": 120}]'
    result = await chart_tools.generate_and_save_chart(
        data_json=data_json,
        chart_type="line",
        x_column="month",
        y_column="sales",
        user_context=user_profile
    )
    assert result.startswith("charts/test_user/chart_")

@pytest.mark.asyncio
@patch('shared_tools.doc_summarizer._initialize_llm', new_callable=AsyncMock)
@patch('shared_tools.doc_summarizer._llm_instance.generate_content', new_callable=AsyncMock)
async def test_summarize_document(mock_generate_content, mock_initialize_llm, user_profile):
    mock_generate_content.return_value = "This is a summary."
    result = await summarize_document("This is a long text.", user_token=user_profile.user_id)
    assert result == "This is a summary."

@patch('pandas.DataFrame.to_csv')
def test_export_data_to_csv(mock_to_csv):
    data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    result = export_data_to_csv(data, "test_user", "test_section")
    assert result.startswith("exports/test_user/test_section/test_section_")

@patch('shared_tools.python_interpreter_tool.PythonInterpreter.run')
def test_python_interpreter_tool(mock_run):
    mock_run.return_value = "4"
    interpreter_tool = PythonInterpreterTool(log_event=None)
    result = interpreter_tool.run_code("print(2+2)")
    assert "Result: 4" in result

@patch('shared_tools.scrapper_tool.requests.get')
def test_scrape_web(mock_get, user_profile):
    mock_get.return_value.text = "<html><body><div class='g'><h3>Title</h3><a href='http://example.com'></a><div class='VwiC3b'>Snippet</div></div></body></html>"
    mock_get.return_value.status_code = 200
    result = scrape_web("test query", user_token=user_profile.user_id)
    assert "Title: Title" in result

"""
Unit tests for the Data Extraction module.

Validates the error handling and retry mechanisms of the BaseScraper class
using mocked HTTP responses to prevent actual network calls during testing.
"""
from unittest.mock import patch, MagicMock
from requests.exceptions import RequestException

import pytest
from bs4 import BeautifulSoup

from src.extract.base_scraper import BaseScraper


class TestScraper(BaseScraper):
    """
    Concrete implementation of BaseScraper purely for testing the base methods.
    """
    def extract_data(self, html_content):
        return [{"dummy": "data"}]
        
    def run(self):
        pass


@pytest.fixture
def scraper():
    """Fixture to supply a fresh TestScraper instance to test sequences."""
    return TestScraper("http://test-url.com")


@patch("src.extract.base_scraper.requests.Session.get")
def test_fetch_page_success(mock_get, scraper):
    """
    Test correct HTML fetch implementation and successful return on HTTP 200 OK.
    
    Time Complexity: O(1) test execution time.
    """
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "<html><body>Test Page</body></html>"
    # Overwrite the built-in raise_for_status to not throw errors on successful mock
    mock_response.raise_for_status = MagicMock()
    
    mock_get.return_value = mock_response

    result = scraper._fetch_page("http://test-url.com")
    
    assert result == "<html><body>Test Page</body></html>"
    mock_get.assert_called_once()


@patch("src.extract.base_scraper.requests.Session.get")
def test_fetch_page_retry_exhaustion(mock_get, scraper):
    """
    Test the robust retry mechanism's failure branch on repeated networking issues.
    
    Time Complexity: O(R) where R is mock retry count (fast due to low delay).
    """
    mock_get.side_effect = RequestException("Connection Refused")

    # Set retries=2, delay=0.01 to speed up test execution
    result = scraper._fetch_page("http://test-url.com", retries=2, delay=0.01)
    
    assert result is None
    assert mock_get.call_count == 2


def test_parse_html_valid(scraper):
    """
    Test that the parser safely builds and utilizes the BeautifulSoup object.
    
    Time Complexity: O(L) bounded by extremely short mock HTML length L.
    """
    html = "<div><span class='price'>999</span></div>"
    soup = scraper._parse_html(html)
    
    assert isinstance(soup, BeautifulSoup)
    price_tag = soup.find('span', class_='price')
    assert price_tag is not None
    assert price_tag.text == "999"


def test_parse_html_empty(scraper):
    """
    Test parser gracefully handles empty inputs.
    """
    assert scraper._parse_html("") is None

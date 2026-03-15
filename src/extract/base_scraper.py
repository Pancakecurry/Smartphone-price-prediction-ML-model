"""
Base Scraper abstraction for the Smartphone Price Prediction project.

This module defines the foundational `BaseScraper` class containing
reusable components for HTTP requests, error handling, and basic HTML parsing.
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import time

import requests
from bs4 import BeautifulSoup
from requests.exceptions import RequestException

from src.logger import get_logger
from src.config import REQUEST_TIMEOUT, USER_AGENT

logger = get_logger(__name__)

class BaseScraper(ABC):
    """
    Abstract base class for all website-specific scrapers.
    
    Provides common networking capabilities, retry mechanisms, and standard
    error logging.
    """
    
    def __init__(self, base_url: str):
        """
        Initialize the scraper with a base URL and standard headers.
        
        Args:
            base_url (str): The primary domain or targeted endpoint.
        """
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        })
        
    def _fetch_page(self, url: str, retries: int = 3, delay: int = 2) -> Optional[str]:
        """
        Fetch HTML content from a given URL with transient error handling.
        
        Args:
            url (str): The target URL to download.
            retries (int): The number of times to retry failed requests.
            delay (int): The wait time between retry attempts (in seconds).
            
        Returns:
            Optional[str]: The raw HTML content as a string, or None if all retries fail.
            
        Time Complexity: O(R) where R is the number of connection attempts.
        Space Complexity: O(N) where N is the length of the string response body in memory.
        """
        for attempt in range(1, retries + 1):
            try:
                response = self.session.get(url, timeout=REQUEST_TIMEOUT)
                response.raise_for_status()
                logger.debug(f"Successfully fetched {url} on attempt {attempt}")
                return response.text
                
            except RequestException as e:
                logger.warning(f"Attempt {attempt}/{retries} failed for {url}: {e}")
                if attempt < retries:
                    time.sleep(delay)
                else:
                    logger.error(f"Failed to fetch {url} after {retries} attempts.")
                    return None

    def _parse_html(self, html_content: str) -> Optional[BeautifulSoup]:
        """
        Parse raw HTML content into a BeautifulSoup object.
        
        Args:
            html_content (str): Raw HTML string.
            
        Returns:
            Optional[BeautifulSoup]: Parsed tree, or None if parsing fails.
            
        Time Complexity: O(L) where L is the length of the HTML document.
        Space Complexity: O(T) where T is the size of the resulting syntax tree.
        """
        if not html_content:
            return None
        try:
            return BeautifulSoup(html_content, 'html.parser')
        except Exception as e:
            logger.error(f"Error parsing HTML content: {e}")
            return None

    @abstractmethod
    def extract_data(self, html_content: str) -> List[Dict[str, Any]]:
        """
        Extract smartphone data from raw HTML. Must be implemented by subclasses.
        
        Args:
            html_content (str): Raw HTML payload.
            
        Returns:
            List[Dict[str, Any]]: A list of dictionaries representing extracted smartphone records.
        """
        pass
        
    @abstractmethod
    def run(self) -> None:
        """
        Execute the full scraping workflow. Must be implemented by subclasses.
        """
        pass

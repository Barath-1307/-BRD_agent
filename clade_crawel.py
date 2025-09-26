# ----------------------------------------------------------------------
# File: website_crawler.py
#
# Purpose:
#   Asynchronously crawls documentation pages on learn.microsoft.com
#   (restricted to English docs under /en-us/) to extract geofencing-
#   related content. Handles both static and dynamic pages using
#   Playwright with Chromium.
#
# Key Features:
#   - Starts from a given URL and crawls up to `max_pages` pages.
#   - Ensures only valid, same-domain, English URLs are followed.
#   - Skips unwanted file types (PDFs, images, media, archives, etc.).
#   - Extracts and cleans text content using BeautifulSoup.
#   - Collects outgoing links for further crawling.
#   - Saves results in both JSON (structured) and TXT (readable) formats.
#   - Maintains logs of visited, failed, and successfully crawled pages.
#   - Generates a crawl summary file with metadata.
#
# Usage:
#   python website_crawler.py
#
#   (async entrypoint runs with asyncio; adjust parameters in `main()`).
#
# Example:
#   crawler = WebsiteCrawler(
#       start_url="https://learn.microsoft.com/en-us/dynamics365/field-service/mobile/configure-geofencing",
#       max_pages=500,
#       delay=1,
#       output_dir="crawled_doc",
#       use_headless=True
#   )
#   await crawler.crawl_website()
#
# Output:
#   - Saves crawled page content into `crawled_doc/` as .json and .txt
#   - Creates `crawl_summary.json` with crawl stats.
# ----------------------------------------------------------------------


import asyncio
import time
import re
import logging
from urllib.parse import urljoin, urlparse, urldefrag
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from collections import deque
import json
import os
from typing import Set, Dict, List, Optional

class WebsiteCrawler:
    def __init__(self, 
                 start_url: str,
                 max_pages: int = 1000,
                 delay: float = 1.0,
                 output_dir: str = "crawled_content",
                 use_headless: bool = True,
                 max_concurrent: int = 3):
        self.start_url = start_url
        self.base_domain = urlparse(start_url).netloc
        self.max_pages = max_pages
        self.delay = delay
        self.output_dir = output_dir
        self.use_headless = use_headless
        self.max_concurrent = max_concurrent
        
        self.visited_urls: Set[str] = set()
        self.failed_urls: Set[str] = set()
        self.url_queue = deque([start_url])
        self.crawled_content: Dict[str, Dict] = {}
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        os.makedirs(output_dir, exist_ok=True)

        self.skip_extensions = {
            '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
            '.zip', '.rar', '.tar', '.gz', '.7z',
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp',
            '.mp3', '.wav', '.mp4', '.avi', '.mov', '.wmv',
            '.css', '.js', '.json', '.xml', '.rss'
        }

    def is_valid_url(self, url: str) -> bool:
        """Check if URL is valid for crawling"""
        if not url or url in self.visited_urls or url in self.failed_urls:
            return False
        
        parsed = urlparse(url)
        
        # Only crawl learn.microsoft.com
        if parsed.netloc != 'learn.microsoft.com':
            return False
        
        # Only English docs (/en-us/)
        if not parsed.path.startswith('/en-us/'):
            return False
        
        # Skip unwanted file types
        if any(url.lower().endswith(ext) for ext in self.skip_extensions):
            return False
        
        skip_patterns = [
            r'/api/', r'/search', r'/feedback', r'/profile', r'/signin'
        ]
        if any(re.search(pattern, url.lower()) for pattern in skip_patterns):
            return False
        
        return True

    def extract_text_content(self, html: str, url: str) -> str:
        """Extract clean text from page"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove non-content parts
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()
            for tag in soup.select('.breadcrumb, .page-actions, .contributors, .feedback-section, .right-rail'):
                tag.decompose()
            
            main_content = soup.select_one('main, .content, article, .markdown')
            text = main_content.get_text(" ", strip=True) if main_content else soup.get_text(" ", strip=True)
            return text
        except Exception as e:
            self.logger.error(f"Error extracting text from {url}: {e}")
            return ""

    def extract_links(self, html: str, current_url: str) -> List[str]:
        """Extract links from page"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            links = []
            for link in soup.find_all('a', href=True):
                href = link['href'].strip()
                if href:
                    absolute_url = urljoin(current_url, href)
                    absolute_url = urldefrag(absolute_url)[0]
                    links.append(absolute_url)
            return links
        except Exception as e:
            self.logger.error(f"Error extracting links from {current_url}: {e}")
            return []

    async def crawl_page(self, browser, url: str) -> Optional[Dict]:
        """Crawl single page with Playwright (handles dynamic content)"""
        context, page = None, None
        try:
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/91.0.4472.124 Safari/537.36"
            )
            page = await context.new_page()
            page.set_default_timeout(40000)
            
            self.logger.info(f"Navigating to {url}")
            response = await page.goto(url, wait_until='networkidle', timeout=40000)
            if not response or response.status >= 400:
                self.logger.warning(f"Bad response for {url}")
                return None
            
            # Extra wait to let dynamic JS sections load
            await page.wait_for_timeout(5000)
            
            final_url = page.url
            html = await page.content()
            title = await page.title()
            text_content = self.extract_text_content(html, final_url)
            links = self.extract_links(html, final_url)
            
            return {
                'original_url': url,
                'final_url': final_url,
                'title': title,
                'text_content': text_content,
                'links': links,
                'timestamp': time.time()
            }
        except Exception as e:
            self.logger.error(f"Error crawling {url}: {e}")
            self.failed_urls.add(url)
            return None
        finally:
            if page:
                await page.close()
            if context:
                await context.close()

    async def save_content(self, url: str, content_data: Dict):
        """Save page content"""
        try:
            parsed = urlparse(url)
            filename = parsed.path.replace('/', '_').strip('_') or "index"
            filename = re.sub(r'[^\w\-_.]', '', filename)[:100]
            
            filepath = os.path.join(self.output_dir, f"{filename}.json")
            counter = 1
            while os.path.exists(filepath):
                filepath = os.path.join(self.output_dir, f"{filename}_{counter}.json")
                counter += 1
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(content_data, f, ensure_ascii=False, indent=2)
            
            with open(filepath.replace('.json', '.txt'), 'w', encoding='utf-8') as f:
                f.write(f"Title: {content_data.get('title', '')}\n")
                f.write(f"URL: {content_data['final_url']}\n")
                f.write("-" * 50 + "\n\n")
                f.write(content_data['text_content'])
        except Exception as e:
            self.logger.error(f"Error saving {url}: {e}")

    async def crawl_website(self):
        """Main crawl loop"""
        self.logger.info(f"Starting crawl at {self.start_url}")
        browser = None
        try:
            async with async_playwright() as playwright:
                browser = await playwright.chromium.launch(
                    headless=self.use_headless,
                    args=['--no-sandbox', '--disable-dev-shm-usage']
                )
                crawled_count = 0
                while self.url_queue and crawled_count < self.max_pages:
                    current_url = self.url_queue.popleft()
                    if not self.is_valid_url(current_url):
                        continue
                    self.visited_urls.add(current_url)
                    content_data = await self.crawl_page(browser, current_url)
                    if content_data:
                        await self.save_content(current_url, content_data)
                        self.crawled_content[current_url] = content_data
                        for link in content_data['links']:
                            if self.is_valid_url(link):
                                self.url_queue.append(link)
                        crawled_count += 1
                    if self.delay > 0:
                        await asyncio.sleep(self.delay)
        finally:
            if browser:
                await browser.close()
            summary = {
                'start_url': self.start_url,
                'total_pages_crawled': len(self.crawled_content),
                'failed_urls': list(self.failed_urls),
                'crawled_urls': list(self.crawled_content.keys()),
                'timestamp': time.time()
            }
            with open(os.path.join(self.output_dir, 'crawl_summary.json'), 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

# Usage
async def main():
    crawler = WebsiteCrawler(
        start_url="https://learn.microsoft.com/en-us/dynamics365/field-service/mobile/configure-geofencing",
        max_pages=500,
        delay=1,
        output_dir="crawled_doc",
        use_headless=True
    )
    await crawler.crawl_website()

if __name__ == "__main__":
    asyncio.run(main())



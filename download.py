import os
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

DOWNLOAD_DIR = "raitamitra_pdfs"
BASE_URL = "https://raitamitra.karnataka.gov.in/info-2/Services/en"
DOMAIN = "https://raitamitra.karnataka.gov.in"

os.makedirs(DOWNLOAD_DIR, exist_ok=True)

async def download_pdf(session, url, filename):
    try:
        async with session.get(url) as resp:
            if resp.status == 200:
                path = os.path.join(DOWNLOAD_DIR, filename)
                with open(path, "wb") as f:
                    f.write(await resp.read())
                print(f" Downloaded: {filename}")
            else:
                print(f"Failed ({resp.status}) for: {url}")
    except Exception as e:
        print(f"Error: {e} for URL: {url}")

async def scrape_and_download():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(BASE_URL)

        # Step 1: Click all dropdowns and extract their links
        dropdown_links = []
        dropdowns = await page.locator(".dropdown .dropbtn").all()
        for dropdown in dropdowns:
            try:
                await dropdown.click()
                await page.wait_for_selector(".dropdown-content a", timeout=2000)
                links = await page.locator(".dropdown-content a").all()
                for link in links:
                    href = await link.get_attribute("href")
                    if href:
                        dropdown_links.append(href)
            except:
                continue

        # Step 2: Get direct links outside dropdowns (like in side nav or other li's)
        all_anchors = await page.locator("ul li a").all()
        direct_links = []
        for a in all_anchors:
            href = await a.get_attribute("href")
            if href and "/info-2/" in href:
                direct_links.append(href)

        # Step 3: Combine and deduplicate all links
        all_links = list(set(dropdown_links + direct_links))

        async with aiohttp.ClientSession() as session:
            for href in all_links:
                if not href:
                    continue
                full_url = DOMAIN + href if href.startswith("/") else href
                print(f"\n Visiting: {full_url}")
                await page.goto(full_url)

                try:
                    await page.wait_for_selector("table.table-striped", timeout=5000)
                    html = await page.content()
                    soup = BeautifulSoup(html, "html.parser")
                    table = soup.find("table", class_="table-striped")

                    if not table:
                        print("No table found.")
                        continue

                    for row in table.find_all("tr")[1:]:
                        cells = row.find_all("td")
                        if len(cells) >= 7:
                            pdf_tag = cells[6].find("a")
                            if pdf_tag:
                                pdf_url = pdf_tag.get("href")
                                if pdf_url and pdf_url.endswith(".pdf"):
                                    filename = pdf_url.split("/")[-1]
                                    await download_pdf(session, pdf_url, filename)

                except Exception as e:
                    print(f"Error while processing {full_url}: {e}")

        await browser.close()

# Run it
asyncio.run(scrape_and_download())

from bs4 import BeautifulSoup
import httpx

url = "https://www.aph.gov.au/Parliamentary_Business/Committees/Senate/Environment_and_Communications/CRA48P/Submissions"


def get_pdf_links():
    res = httpx.get(url)
    soup = BeautifulSoup(res.text, "html.parser")
    table = soup.find("table", {"class": "rgMasterTable"})
import requests
from bs4 import BeautifulSoup
import pandas as pd
import json


def download_sp500_companies():
    # URL of the S&P 500 companies list
    url = 'https://stockanalysis.com/list/sp-500-stocks/'

    # Headers to mimic a browser request
    headers = {
        'User-Agent':
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        # Send GET request to the URL
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find the table containing company data
        table = soup.find('table')

        if table:
            # Extract headers
            headers = []
            for th in table.find_all('th'):
                headers.append(th.text.strip())

            # Extract rows
            rows = []
            for tr in table.find_all('tr')[1:]:  # Skip header row
                row = []
                for td in tr.find_all('td'):
                    row.append(td.text.strip())
                if row:
                    rows.append(row)

            # Create DataFrame
            df = pd.DataFrame(rows, columns=headers)

            # Save to CSV
            df.to_csv('sp500_companies.csv', index=False)
            print(
                f"Successfully downloaded {len(df)} S&P 500 companies to sp500_companies.csv"
            )

            # Save to JSON
            companies_dict = df.to_dict('records')
            with open('sp500_companies.json', 'w') as f:
                json.dump(companies_dict, f, indent=2)
            print("Also saved companies list to sp500_companies.json")

            return df

    except Exception as e:
        print(f"Error downloading S&P 500 companies list: {str(e)}")
        return None


if __name__ == "__main__":
    download_sp500_companies()

import abc
import sys
import gc

import numpy as np
import requests
import sqlite3 as sql
import pandas as pd
import zipfile
from io import BytesIO

from bs4 import BeautifulSoup
from datetime import datetime


class HandlerInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return hasattr(subclass, 'update') and callable(subclass.update) \
            and hasattr(subclass, 'get_result') and callable(subclass.get_result)

    @abc.abstractmethod
    def update(self, df: pd.DataFrame):
        raise NotImplemented

    @abc.abstractmethod
    def get_result(self) -> pd.DataFrame:
        raise NotImplemented


class BaseHandler(HandlerInterface):
    def __init__(self):
        self.__buffer = []

    def update(self, df: pd.DataFrame):
        self.__buffer.append(df)

    def get_result(self) -> pd.DataFrame:
        return pd.concat(self.__buffer)


class DatabaseHandler(HandlerInterface):
    def __init__(self, con: sql.Connection, table_name: str):
        self.__con = con
        self.__table_name = table_name

    def update(self, df: pd.DataFrame):
        df.to_sql(name=self.__table_name, con=self.__con, if_exists='append', index=False)

    def get_result(self) -> pd.DataFrame:
        return None  # pd.read_sql(f'''SELECT * FROM {self.__table_name}''', con=self.__con)


class DownloadHistData:
    __BASE_URL = 'https://www.histdata.com/download-free-forex-historical-data/?/ascii/'
    __TICK_EXTENSION = 'tick-data-quotes/'
    __OHLC_EXTENSION = '1-minute-bar-quotes/'
    __POST_URL = 'https://www.histdata.com/get.php'

    def __init__(self) -> None:
        self.session = requests.Session()

    def tick_url(self) -> str:
        return self.__BASE_URL + self.__TICK_EXTENSION

    @staticmethod
    def tick_columns() -> list[str]:
        return ['date', 'bid', 'ask', 'amount']

    def OHLC_url(self) -> str:
        return self.__BASE_URL + self.__OHLC_EXTENSION

    @staticmethod
    def OHLC_columns() -> list[str]:
        return ['date', 'open', 'high', 'low', 'close', 'volume']

    @property
    def tick_type(self):
        return self.tick_url(), self.tick_columns()

    @property
    def OHLC_type(self):
        return self.OHLC_url(), self.OHLC_columns()

    @staticmethod
    def header(referer=None):
        header = {"User-Agent": "Mozilla/5.0",
                  "Accept-Encoding": None,
                  "Referer": referer}
        return header

    @staticmethod
    def unzip(content) -> pd.DataFrame:
        zip_file = BytesIO(content)
        files = zipfile.ZipFile(zip_file)
        df = pd.read_csv(files.open(files.namelist()[0]), delimiter=',')

        return df

    @staticmethod
    def get_form(resp) -> dict:
        soup = BeautifulSoup(resp.content, 'html5lib')

        file_down = soup.find('form', {'id': 'file_down'})

        form_dict = dict()
        for field in file_down.find_all('input'):
            form_dict.update({field.attrs['name']: field.attrs['value']})

        return form_dict

    @staticmethod
    def date_formatter(string: str):
        return datetime(year=int(string[:4]), month=int(string[4:6]), day=int(string[6:8]),
                        hour=int(string[9:11]), minute=int(string[11:13]), second=int(string[13:15]),
                        microsecond=int(string[15:]) * 1_000)

    def check_years(self, url: str, years: list[int] | int) -> list[int]:
        resp = self.session.get(url=url)

        soup = BeautifulSoup(resp.content, 'html5lib')
        soup = soup.find('div', {'id': 'content'})

        result = []
        for year in soup.find_all('td'):
            year_int = int(year.text)

            if years == -1 or year_int in years:
                result.append(year_int)

        return result

    def check_month(self, url: str, months: list[int] | int) -> list[int]:
        resp = self.session.get(url=url)

        soup = BeautifulSoup(resp.content, 'html5lib')
        soup = soup.find('div', {'id': 'content'})

        result = []
        for month in soup.find_all('a'):
            if 'title' in month.attrs:

                month = month.text.split(sep='-')[0]
                month = month.split(sep='/')[-1]
                month = month.strip()

                month_int = datetime.strptime(month, '%B').month

                if months == -1 or month_int in months:
                    result.append(month_int)

        return result

    def get_labels(self, url: str) -> dict:
        resp = self.session.get(url=url)

        if resp.status_code != requests.codes.ok:
            return dict()

        soup = BeautifulSoup(resp.content, 'html5lib')

        table = soup.find('table')

        labels = {}
        for row in table.find_all('td'):
            label = row.text[:3] + row.text[4:7]
            since = datetime.strptime(row.text[8:-1], '%Y/%B')

            labels.update({label: {'name': label, 'since': np.datetime64(since)}})

        return labels

    def download_label(self, tickr_symbole: str, years: list[int] | int, months: list[int] | int, type_,
                       handler: HandlerInterface) -> pd.DataFrame:
        url_symbole = type_[0] + tickr_symbole + '/'

        years = self.check_years(url=url_symbole, years=years)
        years = sorted(years)

        i_max, i = len(years) * 12, 0
        for year in years:
            url_year = url_symbole + str(year) + '/'

            months = self.check_month(url=url_year, months=months)
            months = sorted(months)

            for moth in months:
                sys.stdout.write(f'\rDownload: {tickr_symbole} [{(i / i_max) * 100:.2f}|100]')
                sys.stdout.flush()

                url_month = url_year + str(moth) + '/'
                resp = self.session.get(url=url_month)

                form_dict = self.get_form(resp=resp)

                resp = self.session.post(self.__POST_URL, form_dict, stream=True,
                                         headers=self.header(referer=url_month))

                df = self.unzip(resp.content)
                df.columns = type_[1]

                df['date'] = df['date'].apply(self.date_formatter)

                handler.update(df=df)

                i += 1

        sys.stdout.write(f'\rDownload: {tickr_symbole} [100|100]\n')
        sys.stdout.flush()

        return handler.get_result()


def get_all_tables(connection: sql.Connection) -> list:
    cursor = connection.cursor()
    cursor.execute('''SELECT * FROM sqlite_master WHERE type=\'table\' ''')

    result = [row[1] for row in cursor.fetchall()]

    cursor.close()

    return result


def main() -> None:
    downloader = DownloadHistData()
    labels = downloader.get_labels('https://www.histdata.com/download-free-forex-data/?/ascii/tick-data-quotes')

    my_symbol = ['EUR', 'USD', 'GBP', 'CHF', 'JPY']

    con = None
    try:
        con = sql.connect(database='ForexData.db')

        all_tables = get_all_tables(con)

        for label in labels:
            first = label[:3]
            second = label[-3:]

            table_name = f'{first}{second}_tick'
            if table_name not in all_tables:
                if first in my_symbol and second in my_symbol:
                    handler = DatabaseHandler(con=con, table_name=table_name)

                    downloader.download_label(tickr_symbole=first + second,
                                              years=[2023], months=list(range(1, 13)),
                                              type_=downloader.tick_type,
                                              handler=handler)

        for table in all_tables:
            df = pd.read_sql(sql=f'''SELECT * FROM {table}''', con=con)
            df.set_index(pd.DatetimeIndex(df['date']), drop=True, inplace=True)
            df.drop(columns=['date'], inplace=True)

            for freq in ['1T', '5T', '15T', '30T', '1H']:
                df_groupby_freq = df.groupby(pd.Grouper(freq=freq))

                df_ohlc = pd.DataFrame()

                df_ohlc['open'] = df_groupby_freq['bid'].first()
                df_ohlc['high'] = df_groupby_freq['bid'].max()
                df_ohlc['low'] = df_groupby_freq['bid'].min()
                df_ohlc['close'] = df_groupby_freq['bid'].last()
                df_ohlc['amount'] = df_groupby_freq['bid'].count()

                df_ohlc['mean'] = df_groupby_freq['bid'].mean()
                df_ohlc['median'] = df_groupby_freq['bid'].median()
                df_ohlc['std'] = df_groupby_freq['bid'].std()

                df_ohlc['date'] = df_ohlc.index

                df_ohlc = df_ohlc[['date', 'open', 'high', 'low', 'close', 'amount', 'mean', 'median', 'std']]

                df_ohlc.to_sql(name=f"{table.split(sep='_')[0]}_ohlc{freq}", con=con, if_exists='fail', index=False)

                del df_ohlc, df_groupby_freq
                gc.collect()

            del df
            gc.collect()

    except Exception as e:
        print(e)

    finally:
        if con:
            con.close()

    pass


if __name__ == '__main__':
    main()

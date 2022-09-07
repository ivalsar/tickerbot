
import pandas as pd
import datetime as dtm
import market_data.fmp as fmp
from image_processing import make_image_from_plotly_figure
from telegraph_api import TelegraphPost
import settings
import formatting

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import numpy as np
from ticker_search import info_dict
import buttons

from posts.base import BaseRequest, PostResponse


class TitledDataFrame(pd.DataFrame):
    title = None
    data = None
    table_spec = None
    highlight_row = None
    text_color = None


class Section(object):
    def __init__(self, title):
        self.title = title
        self.tables = []

    def __str__(self):
        return f"Section: {self.title}, Num Tables: {len(self.tables)}"

    def get_ratios_ttm(ticker):
        return fmp.get(f'ratios-ttm/{ticker}')[0]


def make_plotly_figure(section):

    row1_color = sns.color_palette('Greys', 8).as_hex()[0]
    row2_color = sns.color_palette('Blues', 8).as_hex()[0]
    head_color = sns.color_palette('Blues', 8).as_hex()[-2]

    row_height = 30
    spacing = 10
    row_width = 90
    index_width = 180

    heights = [(t.shape[0] + 1) * row_height for t in section.tables]
    total_height = sum(heights) + spacing * (len(section.tables) - 1)

    num_tables = len(section.tables)
    fig = make_subplots(
        rows=num_tables, cols=1,
        shared_xaxes=True,
        vertical_spacing=spacing / total_height,
        specs=[[{"type": "table"}]] * num_tables,
        row_heights=heights
    )

    for table_num in range(num_tables):
        table = section.tables[table_num]
        table_frame = table.copy()
        table_frame = table_frame.reset_index()
        table_frame['index'] = table_frame['index'].apply(lambda c: f'<b>{c}')

        table_columns = [f'<b>{c}' for c in [table.title] + table_frame.columns.tolist()[1:]]
        table_data = table_frame.values.T

        #     colors = [['black']*table_data.shape[0]] + table.text_color
        #     colors = np.array(colors)

        plotly_table = go.Table(
            name='Bla',
            columnwidth=[index_width, row_width],
            header=dict(values=table_columns,
                        fill_color=head_color,
                        font=dict(color='white', family="Arial", size=12),
                        align=['left', 'center'],
                        height=row_height),
            cells=dict(values=table_data,
                       fill_color=[np.where(table.highlight_row, row2_color, row1_color)],
                       font=dict(color=np.array(table.text_color).T, family="Arial", size=12),
                       align=['left', 'right'],
                       height=row_height))

        fig.add_trace(plotly_table, row=table_num + 1, col=1)

    fig.update_layout(height=total_height,
                      width=index_width + row_width * len(table_columns),
                      margin=go.layout.Margin(l=0, r=0, b=0, t=0))

    return fig


def create_report_sections(df, fa_spec, is_annual, num_periods):

    sections = []

    if is_annual:
        years = df.iloc[:, :num_periods].T['date'].apply(lambda x: dtm.datetime.strptime(x, "%Y-%m-%d").year)
        header = [f'FY {y}' for y in years]
    else:
        header = ['FQ X' + (f'-{p}' if p > 0 else '') for p in range(num_periods)]

    for section_name in fa_spec.section.unique():

        section = Section(section_name)
        sections.append(section)

        for table_name in fa_spec[fa_spec.section == section_name].table.unique():
            table_spec = fa_spec[fa_spec.table.eq(table_name)]

            table_df = pd.DataFrame(columns=list(range(num_periods)))

            highlights = []
            text_color = []

            for i, row in table_spec.iterrows():
                vals = df.loc[(row.endpoint, row.field), :]

                # fix required here
                # if row.fx_adjust:
                #     vals = vals / df.loc['fx_rate'].values[0]

                row_series = vals.iloc[:num_periods].copy()
                row_series.name = row.row_name

                if row.format == 'PERCENT':
                    row_series.iloc[:] = formatting.format_percents(row_series.values * 100, False)
                elif row.format == 'RATIO':
                    row_series.iloc[:] = formatting.format_ratios(row_series.values)
                elif row.format in ['U', 'K', 'M', 'B', 'T']:
                    row_series.iloc[:] = formatting.format_values(row_series.values, row.format)

                if row.highlight:
                    row_series.iloc[:] = [f'<b>{v}' for v in row_series.values]

                table_df = table_df.append(row_series)

                highlights.append(row.highlight | row.calc_growth)
                text_color.append(['black'] * (table_df.shape[1] + 1))

                if row.calc_growth:
                    growth_col_name = '<i>       growth %'
                    growth_vals = (vals.iloc[:num_periods] / vals.values[1:num_periods + 1] - 1) * 100
                    row_series = growth_vals.copy()
                    row_series.name = growth_col_name
                    row_series.iloc[:] = [f'<i>{v}' for v in formatting.format_percents(row_series.values, True)]

                    table_df = table_df.append(row_series)

                    highlights.append(False)
                    text_color.append(['black'] + np.where(growth_vals >= 0, 'green', 'red').tolist())

            table_df = TitledDataFrame(table_df)
            table_df.title = table_name
            table_df.columns = header
            table_df.table_spec = table_spec
            table_df.highlight_row = highlights
            table_df.text_color = text_color

            section.tables.append(table_df)

    return sections


class FaRequest(BaseRequest):

    request_name = 'fa'
    request_params = ['ticker', 'period']

    def make(self, ticker=None, period=None):

        if ticker is None:
            return PostResponse(text='Please specify ticker')

        if period is None:
            menu_data = [
                        ("Annual", f"/fa {ticker} A"),
                        ("Quarterly", f"/fa {ticker} Q")
                    ]
            reply_markup = buttons.build_menu(menu_data, 2)

            return PostResponse(text="_Select report type_", reply_markup=reply_markup)

        # all parameters are present
        num_periods = 5

        is_annual = period[0] in ['Y', 'A']

        df = fmp.get_financials(ticker, is_annual, limit=num_periods+1)

        df = df.reset_index()
        df[('other', 'numShares')] = df[('enterprise-values', 'enterpriseValue')] \
                                     / df[('historical-discounted-cash-flow-statement', 'price')]

        df[('historical-discounted-cash-flow-statement', 'dcf')] *= df[('other', 'numShares')]

        df = df.T
        df.loc[('other', 'industryPE'), :] = None
        df.loc['fx_rate', :] = 1

        sec_pe = fmp.get_sector_pe(ticker)

        df.loc[('other', 'industryPE')].iloc[0] = sec_pe

        df = df.fillna('')

        fa_spec = pd.read_csv(settings.FA_SPEC_CSV)

        sections = create_report_sections(df, fa_spec, is_annual, num_periods)

        titles = []
        image_files = []
        for section in sections:
            fig = make_plotly_figure(section)
            image_file = make_image_from_plotly_figure(fig)

            titles.append(section.title)
            image_files.append(image_file)

        # make the Telegraph post
        p = TelegraphPost(f'Fundamental Analysis: {info_dict[ticker]["name"]}')
        images = p.upload_files(image_files)

        for section_title, image in zip(titles, images):
            p.add_subtitle_node(section_title)
            p.add_url_image_node(image)

        post_url = p.post()

        return PostResponse(text=post_url)




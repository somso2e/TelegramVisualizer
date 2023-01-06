from tqdm import tqdm
import plotly.graph_objects as go
import json
import pandas as pd
import plotly
import plotly.io as pio
from plotly.subplots import make_subplots
import typing
import os


class TelegramVisualizer():
    MEDIA_TYPES = {"message": "Messages",
                   "photo": "Photos",
                   "voice_message": "Voice Messages",
                   "audio_file": "Audio Files",
                   "video_message": "Video Messages",
                   "video_file": "Video Files",
                   "sticker": "Stickers",
                   "animation": "Animations(GIF)",
                   }

    def __init__(self,
                 chat_history_path: str,
                 show_pie_charts: bool = True,
                 show_timeseries_plot: bool = True,
                 plot_total_messages_only: bool = False,
                 colors: typing.List[str] = plotly.colors.qualitative.Plotly,
                 template: str = "plotly_dark",
                 timeseries_rolling_average_window: int = 7,
                 dimensions: typing.Tuple[int, int] = None,
                 cache_dataframes: bool = True,
                 restore_from_cache: bool = True,
                 ):
        self.chat_history_path = chat_history_path
        self.dimensions = dimensions
        self.timeseries_rolling_average_window = timeseries_rolling_average_window
        self.colors = colors
        self.plot_total_messages_only = plot_total_messages_only
        self.show_timeseires_plot = show_timeseries_plot
        self.show_pie_charts = show_pie_charts

        self.legend_group = 1

        pio.templates.default = template

        self.dir_name = chat_history_path
        self.messages_df_path = os.path.join(
            self.dir_name, "messages.csv")
        self.media_count_df_path = os.path.join(
            self.dir_name, "media_counts.csv")

        if restore_from_cache:
            if os.path.exists(self.messages_df_path) and os.path.exists(self.media_count_df_path):
                self.timeseries_df = pd.read_csv(
                    self.messages_df_path, index_col="date")
                self.media_count_df = pd.read_csv(
                    self.media_count_df_path, index_col="user")

                self.users = self.media_count_df.index

                print("Data loaded from cached .csv files")
            else:
                print("No cache found. Will read from .json file.")
                self.__generate_df()

        else:
            self.__generate_df()

        if cache_dataframes:
            self.timeseries_df.to_csv(self.messages_df_path)
            self.media_count_df.to_csv(self.media_count_df_path)

        self.generate_plots()

    def __generate_df(self):
        json_path = os.path.join(self.chat_history_path, "result.json")
        if not os.path.exists(json_path):
            raise(ValueError(
                "No result.json file could be found in {} directory".format(self.dir_name)))

        with open(json_path, "r", encoding="utf-8") as f:
            json_file = json.load(f)

        self.users = []
        dates = []
        for msg in json_file["messages"]:
            # Skip Pins notifications
            if msg["type"] != "message":
                continue
            user = msg["from"]
            # If a user was called "Total" rename it to prevent conflicts
            if user == "Total":
                user = "Total_user"

            if user not in self.users:
                self.users.append(user)

            date = msg["date"].split("T")[0]
            if date not in dates:
                dates.append(date)

        self.timeseries_df = pd.DataFrame()
        self.timeseries_df["date"] = dates
        self.timeseries_df.set_index("date", inplace=True)

        for user in self.users:
            self.timeseries_df[user] = 0

        self.media_count_df = pd.DataFrame(
            columns=self.MEDIA_TYPES.values(), index=self.users)
        self.media_count_df.index.name = "user"

        for media_type in self.MEDIA_TYPES.values():
            self.media_count_df[media_type].values[:] = 0

        for msg in tqdm(json_file["messages"], desc="Reading Messages"):
            if msg["type"] != "message":
                continue
            date = msg["date"].split("T")[0]
            user = msg["from"]
            self.timeseries_df.loc[date, user] += 1

            if "media_type" in msg:
                self.media_count_df[self.MEDIA_TYPES[msg["media_type"]]][user] += 1
            if "photo" in msg:
                self.media_count_df["Photos"][user] += 1

        self.media_count_df[self.MEDIA_TYPES["message"]
                            ] = self.timeseries_df[self.users].sum(axis=0)

    def apply_rolling_average(self, df: pd.DataFrame):
        for user in self.users:
            df[user] = df[user].rolling(
                window=self.timeseries_rolling_average_window, min_periods=1).mean()
        return df

    def __get_legend_group(self):
        legend_group = str(self.legend_group)
        self.legend_group += 1
        return legend_group

    def get_timeseries_plot(self, df: pd.DataFrame):
        df["Total"] = df.loc[:, self.users].sum(axis=1).astype(int)

        legendgroup = self.__get_legend_group()
        for i, column in enumerate(df.columns):
            if self.plot_total_messages_only and column != "Total":
                continue
            yield go.Scatter(x=df.index,
                             y=df[column],
                             mode="lines",
                             name=column,
                             legendgroup=legendgroup,
                             marker=dict(color=self.colors[i]))

    def get_media_pie_charts(self, df: pd.DataFrame):
        pie_charts = []
        titles = []
        legendgroup = self.__get_legend_group()

        for media_type in self.MEDIA_TYPES.values():
            if df[media_type].sum() > 0:
                pie_charts.append(
                    go.Pie(labels=df[media_type].index,
                           values=df[media_type].values,
                           legendgroup=legendgroup,
                           name=media_type,
                           hoverinfo="label+value+percent",
                           marker=dict(colors=self.colors)))
                titles.append(media_type)
        return pie_charts, titles

    def generate_plots(self):

        pie_charts, titles = self.get_media_pie_charts(self.media_count_df)

        n_rows = -(-len(pie_charts)//2)

        specs = []
        if self.show_pie_charts:
            if len(pie_charts) % 2 == 0:
                specs = [[{"type": "domain"}, {"type": "domain"}]
                         for _ in range(n_rows)]
            else:
                specs = [[{"type": "domain"}, {"type": "domain"}]
                         for _ in range(n_rows-1)]
                specs.append([{"colspan": 2, "type": "domain"}, None])

        if self.show_timeseires_plot:
            specs.append([{"colspan": 2, "type": "xy"}, None])

        subplot_titles = titles+["Date"]
        self.fig = make_subplots(
            rows=len(specs), cols=2,
            specs=specs,
            subplot_titles=subplot_titles
        )
        column = 1
        row = 1
        if self.show_pie_charts:
            for i, pie_chart in enumerate(pie_charts):
                column = i % 2+1
                row = i // 2+1
                self.fig.add_trace(pie_chart, row=row, col=column)
            row += 1
            column -= 1

        if self.show_timeseires_plot:
            timeseries_df_avg = self.apply_rolling_average(self.timeseries_df)
            # fig.update_traces(marker_cmin=0)
            for timeseries_plot in self.get_timeseries_plot(timeseries_df_avg):
                self.fig.add_trace(timeseries_plot, row=row, col=column)

            self.fig.update_xaxes(
                rangeslider_visible=True,
                rangeslider_thickness=0.05,
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month",
                             stepmode="backward"),
                        dict(count=6, label="6m", step="month",
                             stepmode="backward"),
                        dict(count=1, label="YTD",
                             step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year",
                             stepmode="backward"),
                        dict(step="all")
                    ])
                ),
            )

        if self.dimensions is None:
            width = 1000
            height = 400*len(specs)
        else:
            width = self.dimensions[0]
            width = self.dimensions[1]

        self.fig.update_layout(width=width,
                               height=height,
                               legend_tracegroupgap=height*0.72,
                               xaxis_rangeselector_font_color='black',
                               xaxis_rangeselector_activecolor='#555555',
                               xaxis_rangeselector_bgcolor='#202124',
                               coloraxis=dict(colorscale='RdBu'),
                               )

    def plot(self):
        self.fig.show()

    def save(self, save_path):
        extension = (os.path.splitext(save_path)[-1]).lower()
        if extension == ".html":
            self.fig.write_html(save_path)
        elif extension in [".png", ".jpeg", ".jpg", ".webp", ".svg", ".pdf", ".eps"]:
            pio.kaleido.scope.mathjax = None
            self.fig.write_image(save_path)
        else:
            raise(ValueError("Invalid file format: {}".format(extension)))

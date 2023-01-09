from tqdm import tqdm
import plotly.graph_objects as go
import json
import pandas as pd
import plotly
import plotly.io as pio
from plotly.subplots import make_subplots
import typing
import os
import numpy as np


class TelegramVisualizer():
    MEDIA_TYPES = {"message": "Messages",
                   "photo": "Photos",
                   "file": "Files",
                   "voice_message": "Voice Messages",
                   "audio_file": "Audio",
                   "video_message": "Video Messages",
                   "video_file": "Video",
                   "sticker": "Stickers",
                   "animation": "Animations(GIF)",
                   }

    def __init__(self,
                 chat_history_path: str,
                 plot_pie_charts: bool = True,
                 plot_timeseires: bool = True,
                 plot_total_messages_only: bool = False,
                 timeseries_rolling_average_window: int = 7,
                 image_size: typing.Tuple[int, int] = None,
                 columns: int = 3,
                 colors: typing.List[str] = plotly.colors.qualitative.Plotly,
                 dark_mode: bool = True,
                 export_data: bool = True,
                 restore_from_cache: bool = True,
                 ):
        """
        # TelegramVisualizer

        This allows you to create cool and interesting stats and interactive graphs from your Telegram chats.

        Parameters
        ----------
        chat_history_path : str
            Path of the folder containing the `result.json` file or the file itself 
        plot_pie_charts : bool, optional
            If `True`(default), then pie chart of number of messages and differnt medias will be shown, by default True
        plot_timeseires : bool, optional
            If `True`(default), then the graph of "number of messages by date" will be shown, by default True
        plot_total_messages_only : bool, optional
            If `True`, then only the total messages/date will be shown, by default False
        timeseries_rolling_average_window : int, optional
            Rolling average window in number of days, by default 7 days
        image_size : (int, int), optional
            The size of the image/HTML in pixels. If `None`, then it will be choosen automatically, by default None
        columns: int, optional
            Number of columns for pie charts, by default 3 
        colors : list[str], optional
            A `list` of colors to be used to indicate users, by default plotly.colors.qualitative.Plotly
        dark_mode : bool, optional
            If `True`(default), dark mode by default True
        export_data : bool, optional
            If `True`(default), then `.csv` files are generated from `result.json` file.
            This allows for faster run time for later, by default True
        restore_from_cache : bool, optional
            If `True`(default), then data is loaded from , by default True
        """
        self.chat_history_path = chat_history_path

        self.plot_pie_charts = plot_pie_charts

        self.plot_timeseires = plot_timeseires
        self.plot_total_messages_only = plot_total_messages_only
        self.timeseries_rolling_average_window = timeseries_rolling_average_window

        if plot_pie_charts == False and plot_timeseires == False:
            raise(ValueError(
                "You can't have both pie charts and timeseries graph turned off."))

        self.image_size = image_size
        if columns <= 0:
            raise(ValueError("Number of columns can't be les"))
        self.columns = columns

        self.colors = colors
        self.dark_mode = dark_mode

        if dark_mode:
            pio.templates.default = "plotly_dark"
        else:
            pio.templates.default = "plotly_white"

        self.legend_group = 1
        if os.path.basename(chat_history_path) == "result.json":
            self.dir_name = os.path.dirname(chat_history_path)
        else:
            self.dir_name = chat_history_path
        print(self.dir_name)
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
                self.__generate_df()

        else:
            self.__generate_df()

        if export_data:
            self.timeseries_df.to_csv(self.messages_df_path)
            self.media_count_df.to_csv(self.media_count_df_path)

        self.generate_plots()

    def __generate_df(self):
        json_path = os.path.join(self.dir_name, "result.json")
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
            elif "mime_type" in msg:
                self.media_count_df["Files"][user] += 1
            elif "photo" in msg:
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
                             marker=dict(color=self.colors[i % len(self.colors)]))

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

        if self.plot_pie_charts:
            pie_charts, titles = self.get_media_pie_charts(self.media_count_df)
            n_small_pies = len(pie_charts)-1

            n_columns = min(self.columns, n_small_pies+2)

            msg_pie_dim = (min(n_columns, 2), min(n_columns, 2))
            cells_occupied_by_pies = n_small_pies + \
                msg_pie_dim[0] * msg_pie_dim[1]
            n_rows = int(np.ceil(cells_occupied_by_pies/n_columns)) + \
                self.plot_timeseires

            specs = np.full((n_rows, n_columns), None, dtype=object)

            # spec of the bigger messges pie chart
            specs[:msg_pie_dim[0], :msg_pie_dim[1]
                  ] = "RESERVED FOR MESSAGES PIE"

            specs[0, 0] = {"type": "domain",
                           "rowspan": msg_pie_dim[0],
                           "colspan": msg_pie_dim[1], }
            specs = specs.reshape(-1)
            p = 0
            for i in range(specs.shape[0]):
                if specs[i] is None:
                    specs[i] = {"type": "pie"}
                    p += 1
                if p == n_small_pies:
                    break
            specs[specs == "RESERVED FOR MESSAGES PIE"] = None
            specs = specs.reshape((n_rows, n_columns))

            if self.plot_timeseires:
                specs[-1, 0] = {"colspan": n_columns, "type": "xy"}

        else:
            specs = np.array([[{"type": "xy"}]])

        inds = np.argwhere(specs != None)

        subplot_titles = titles
        self.fig = make_subplots(
            rows=int(n_rows), cols=int(n_columns),
            specs=specs.tolist(),
            subplot_titles=subplot_titles,
            vertical_spacing=0.02,
            horizontal_spacing=0.02,

        )

        if self.plot_pie_charts:

            for i, pie_chart in enumerate(pie_charts):
                row, column = inds[i]
                self.fig.add_trace(pie_chart, row=row+1, col=column+1)

        if self.plot_timeseires:
            timeseries_df_avg = self.apply_rolling_average(self.timeseries_df)
            row, column = inds[-1]
            for timeseries_plot in self.get_timeseries_plot(timeseries_df_avg):
                self.fig.add_trace(timeseries_plot, row=row+1, col=column+1)

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

        if self.image_size is None:
            width = 500 * specs.shape[1]
            height = 500*specs.shape[0]
        else:
            width = self.image_size[0]
            height = self.image_size[1]

        self.fig.update_layout(autosize=False,
                               width=width,
                               height=height,
                               legend_tracegroupgap=height *
                               (n_rows-1)/n_rows*0.9,
                               margin=dict(l=20, r=0, t=20, b=0))

        if self.dark_mode:
            self.fig.update_layout(xaxis_rangeselector_font_color='black',
                                   xaxis_rangeselector_activecolor='#555555',
                                   xaxis_rangeselector_bgcolor='#202124')


    def plot(self):
        self.fig.show()

    def save(self, save_path):
        extension = (os.path.splitext(save_path)[-1]).lower()
        if extension == ".html":
            self.fig.write_html(save_path)
        elif extension in [".png", ".jpeg", ".jpg", ".webp", ".svg", ".pdf", ".eps"]:
            try:
                import kaleido
            except ImportError as e:
                raise(ImportError(
                    "You need the kaleido package to export as static image.\nUse \"pip install kaleido\""))
            self.fig.update_xaxes(rangeselector=None)
            self.fig.write_image(save_path)
        else:
            raise(ValueError("Invalid file format: {}".format(extension)))

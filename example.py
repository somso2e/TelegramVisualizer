from TelegramVisualizer import TelegramVisualizer

telegram_visualizer = TelegramVisualizer(
    chat_history_path="data/",
    columns=3,
    dark_mode=True
)

# telegram_visualizer.plot()
telegram_visualizer.save("output_3_columns_dark_mode.png")
from TelegramVisualizer import TelegramVisualizer

telegram_visualizer = TelegramVisualizer(
    chat_history_path="data/",
)

telegram_visualizer.plot()

telegram_visualizer.save("output.html")
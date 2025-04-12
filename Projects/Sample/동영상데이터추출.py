from pytube import YouTube
from IPython.display import HTML

url = "https://www.youtube.com/watch?v=ILqJOHYYlkc"

yt = YouTube(url)

stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()

stream.download()


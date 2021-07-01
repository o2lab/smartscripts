import hashlib, random, requests
from PIL import Image
from io import BytesIO

def get_url(urls):
    return urls[random.randint(0, len(urls) - 1)]

def get_file(url):
    r = requests.get(url)
    im = Image.open(BytesIO(r.content))
    im.show()

urls_file = open("urls.txt", "r")
urls = [url.rstrip() for url in urls_file.readlines()]
urls_file.close()

get_file(get_url(urls))
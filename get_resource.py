import urllib.request

url = 'https://raw.githubusercontent.com/maskot1977/ipython_notebook/master/toydata/location.txt'

urllib.request.urlretrieve(url, 'location.csv')

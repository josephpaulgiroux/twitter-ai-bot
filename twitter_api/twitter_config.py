from twitter import *
from util import make_sure_path_exists
import os

US_WOE_ID = 23424977
SF_WOE_ID = 2487956

token_path = os.path.expanduser("/home/joe/twitterbot/erictau_credentials.cnf")
# make_sure_path_exists(token_path)

CONSUMER_KEY = "NyTSPrvxwyNZsvXyAcUZYIvbQ"
CONSUMER_SECRET = "te1cpp8mKdAjr2KTfsii2jUy42dECJyH5ewAH0JFoMdQvKGGqj"

ACCESS_TOKEN, ACCESS_TOKEN_SECRET = read_token_file(token_path)

bot = Twitter(auth=OAuth(ACCESS_TOKEN, ACCESS_TOKEN_SECRET, CONSUMER_KEY, CONSUMER_SECRET))


BLACKLIST = [
    u"nigger", u"cunt", u"spic", u"kike", u"nigga", u"twat", u"faggot"
]
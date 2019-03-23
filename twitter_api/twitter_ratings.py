import datetime as dt
from twitter_api.twitter_config import BLACKLIST
from util import *
from bs4 import UnicodeDammit



def evaluate_user(user):
    followers = user["followers"]
    follows = user["friends_count"]
    statuses = user["statuses_count"]
    favorites = user["favourites_count"]


def validate_tweet(tweet):
    if "possibly_sensitive" in tweet and tweet["possibly_sensitive"]:
        return False
    for word in BLACKLIST:
        if word in tweet["text"]:
            return False



def compress_tweet(tweet):
    user = tweet["user"]
    goddamnit = UnicodeDammit(tweet["text"])
    try:
        text = goddamnit.unicode_markup
    except UnicodeEncodeError as ex:
        print ex.message
        print "FUCKING CUNT"
        print tweet["text"]
        goddamnit = UnicodeDammit(tweet["text"])
        print goddamnit
        print goddamnit.unicode_markup
        sys.exit()
    print text
    followers = user["followers_count"]
    if followers < 50 and tweet["user"]["id_str"] != "830578232575406082":
        return None, None
    unique_words, total_words, content_tag_ratio, stemmed_text = stem_and_tokenize(text)
    retweets = tweet["retweet_count"]
    favorites = tweet["favorite_count"]
    created_at = str_to_date(tweet["created_at"])
    hours_old = (dt.datetime.now() - created_at).seconds / 3600.0
    virality = float(retweets*3+1) / followers
    popularity = float(favorites*2+1) / followers
    return tweet["id_str"], dict(user_id=user["id_str"], text=text,
                                stemmed_text=stemmed_text,
                                virality=virality, popularity=popularity,
                                retweets=retweets, favorites=favorites,
                                followers=followers, hours_old=hours_old,
                                created_at=created_at, char_length=len(text),
                                unique_words=unique_words, total_words=total_words,
                                content_tag_ratio=content_tag_ratio)

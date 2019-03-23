

from twitter_api.twitter_config import *
from twitter_api.agent import eric
print eric
from util import *
import time
import random
from MY import *
# eric.get_tweets_from_home_timeline()
# eric.compress_and_store_tweets(eric.tweet_test)

def main():
    # master loop for one iteration of eric's life

    try:
        topics = refresh_trends(eric)
        pass
    except TwitterHTTPError as ex:
        warn("Fucking Twitter:  "+ex.message)
        pass

    try:
        search_topics(eric, topics)
        pass
    except TwitterHTTPError as ex:
        eric.pickle_me()
        warn("Fucking Twitter:  " + ex.message)
        pass
    eric.pickle_me()

    # additional round of tweets
    try:
        additional_tweets(eric, topics)
    except TwitterHTTPError as ex:
        warn("Fucking Twitter:  " + ex.message)
        pass

    eric.pickle_me()



def additional_tweets(eric, topics):
    random.shuffle(topics)
    for topic in topics:
        if topic in eric.topic_memory and len(eric.topic_memory[topic]) > 1000:
            try:
                eric.post_n_random_from_pool(topic, n=1, pool_size=50)
            except KeyError:
                eric.generate_babbler(topic)
                eric.post_n_random_from_pool(topic, n=1, pool_size=50)


def search_topics(eric, topics):
    random.shuffle(topics)
    for topic in topics:
        if topic in eric.topic_memory:
            eric.get_topic_tweets(topic, newer=True)
            eric.get_topic_tweets(topic, newer=False)
            pass
        else:
            eric.topic_memory[topic] = dict()
            eric.get_topic_tweets(topic)
            # seed with about a thousand tweets or so
            for n in xrange(0, 15):
                added = eric.get_topic_tweets(topic, newer=False)
                if not added or added < 5:
                    break

            # look at the "best" tweet in terms of virality and popularity in each topic, ffollow
            # the authors, and favorite their tweets
            best_tweets = []
            for metric in ["popularity", "virality"]:
                best_tweets += eric.get_n_best_tweets(topic, n=1, metric=metric)
            for tweet_id, tweet in best_tweets:
                eric.follow_user(tweet["user_id"])
                eric.favorite_tweet(tweet_id)

        if topic in eric.topic_memory and len(eric.topic_memory[topic]) > 1000:
            eric.generate_babbler(topic)
            eric.post_n_random_from_pool(topic, n=1, pool_size=30)


def refresh_trends(eric):
    # refresh trends
    eric.get_place_trends()
    topics = eric.get_possible_topics().keys()

    expired_trends = []
    for topic in eric.topic_memory.keys():
        if topic not in topics:
            expired_trends.append(topic)
    for topic in expired_trends:
        # archive the data in case we want to draw upon it later
        pickle_me(eric.topic_memory[topic], topic[3:] + ".p")
        # write this into a cute method later
        del eric.topic_memory[topic]
    return topics


if __name__=="__main__":
    main()
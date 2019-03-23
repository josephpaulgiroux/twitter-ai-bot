


from twitter_api.twitter_config import *
from twitter_api.twitter_ratings import *
from util import *
from warnings import warn
from markov_babbler.markov_class import WeightedMarkovBabbler
import numpy as np
import datetime as dt
import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.decomposition import TruncatedSVD
from sknn.mlp import Regressor, Layer
from scipy import stats
from sklearn.grid_search import RandomizedSearchCV
import urllib
from util import PICKLE_PATH


class Agent(object):

    def __init__(self, tw, pickle_file=""):

        self.newest_tweet_seen = ""
        self.oldest_tweet_seen = ""

        self.suggestion_groups = []

        self.suggested_usergroups = []
        self.suggested_users = []

        self.personal_memory = {} # tweets I have made

        self.observational_memory = {} # tweets I've seen and retained
        self.topic_memory = {} # tweets in trending hashtags
        self.topic_babblers = {}

        self.universe = {} # users I know about

        self.tweet_test = []
        self.trend_locations = []
        self.place_trends = []

        if pickle_file:
            self.pickle_file = pickle_file
            try:
                self.__dict__.update(unpickle_me(pickle_file))
            except Exception as ex:
                warn(ex.message)
        self.tw = tw

    def __repr__(self):
        msg = "I have {} users in my known universe.\n".format(len(self.universe))
        msg += "I have {} tweets in my personal memory.\n".format(len(self.personal_memory))
        msg += "I know about the following topics:\n"
        for k, v in self.topic_memory.iteritems():
            msg += "{} ({} tweets)\n".format(k, len(v))
        return msg

    def generate_master_babbler(self):
        self.master_babbler3 = WeightedMarkovBabbler(n_back=3)
        for root, dir, files in os.walk(PICKLE_PATH.format("")):
            for file_name in files:
                if file_name != BRAIN_PICKLE:
                    print "***"
                    print file_name
                    print "***"
                    topic_data = unpickle_me(file_name)
                    documents, weights = self.process_tweets_for_babbler(topic_data)
                    self.master_babbler3.fit(documents, weights)
        for topic, topic_data in self.topic_memory.iteritems():
            print "***"
            print topic
            print "***"
            documents, weights = self.process_tweets_for_babbler(topic_data)
            self.master_babbler3.fit(documents, weights)
        
    def pickle_me(self, pickle_file=None):
        print "Pickle brain in a jar of brine."
        if not pickle_file:
            pickle_file = self.pickle_file
        brain = {k: v for k,v in self.__dict__.iteritems() if k != "tw"}
        pickle_me(brain, pickle_file)
        print "Pickling complete."

    def get_trend_locations(self):
        locations = self.tw.trends.available()
        self.trend_locations = locations
        self.pickle_me()

    def get_place_trends(self, place="US"):
        trends = self.tw.trends.place(_id=SF_WOE_ID if place=="SF" else US_WOE_ID)
        self.place_trends = trends
        print trends
        self.pickle_me()

    def store_suggestion_groups(self):
        sugg = self.tw.users.suggestions()
        self.suggestion_groups.append(sugg)

    def store_all_suggested_users(self):
        suggestion_group = self.suggestion_groups[-1]
        slugs = [item["slug"] for item in suggestion_group]
        suggested_usergroups = dict()
        for slug in slugs:
            print "Getting suggestions for {}".format(slug)
            try:
                suggested_usergroups[slug] = self.get_suggested_users(slug)
            except TwitterHTTPError as ex:
                print "Fucking twitter: {}".format(ex.message)

                break
        self.suggested_usergroups.append(suggested_usergroups)
        self.pickle_me()

    def get_suggested_users(self, slug):
        users = getattr(self.tw.users.suggestions, slug)(slug=slug)
        return users

    def follow_user(self, user_id):
        if user_id in self.universe:
            print "Already following user {}".format(user_id)
            return
        else:
            user = self.tw.friendships.create(user_id=user_id)
            print user_id
            self.universe[user_id] = user

    def favorite_tweet(self, tweet_id):
        try:
            self.tw.favorites.create(_id=tweet_id)
        except TwitterHTTPError:
            print "Already favorited that."

    def seed_followers_from_suggestions(self, refresh_groups=True):
        # get our most recent chunk of suggestions, and pick
        # someone from each slug to follow.  For now, since
        # high level suggestions are big name celebrities and
        # the like, get the ones following the most others
        # within their category, with the hopes of introducing
        # the most diversity into our follow pool.
        if refresh_groups:
            self.store_all_suggested_users()
        suggested_usergroups = self.suggested_usergroups[-1]
        for slug, usergroup in suggested_usergroups.iteritems():
            best_user = max([item for item in usergroup["users"] if item["id_str"] not in self.universe],
                            key=lambda user: user["friends_count"])
            print best_user["name"]
            self.follow_user(best_user["id_str"])
        self.pickle_me()


    def get_tweets_from_home_timeline(self, count=200, since_id=None, max_id=None, **kwargs):

        default_args = dict(count=200, exclude_replies=False)
        if since_id:
            default_args["since_id"] = since_id
        if max_id:
            default_args["max_id"] = max_id
        default_args.update(**kwargs)
        self.tweet_test = self.tw.statuses.home_timeline(**default_args)
        self.compress_and_store_tweets(self.tweet_test)

    def compress_and_store_tweets(self, tweet_data, topic=None, personal=False):
        if not personal and not topic:
            self.pickle_me()
            warn("This should not happen, must designate where to store data.")
        added = 0
        if not personal:
            if topic not in self.topic_memory:
                self.topic_memory[topic] = {}
            for tweet in tweet_data:
                tweet_id, compressed_tweet = compress_tweet(tweet)
                if tweet_id:
                    added += tweet_id not in self.topic_memory[topic]
                    self.topic_memory[topic][tweet_id] = compressed_tweet
                    added += 1
        else:
            for tweet in tweet_data:
                tweet_id, compressed_tweet = compress_tweet(tweet)
                if tweet_id:
                    added += tweet_id not in self.personal_memory
                    self.personal_memory[tweet_id] = compressed_tweet
        return added

    def generate_babbler(self, topic):
        if topic not in self.topic_memory:
            return
        self.topic_babblers[topic] = WeightedMarkovBabbler(n_back=2)
        documents, weights = self.process_tweets_for_babbler(self.topic_memory[topic])
        self.topic_babblers[topic].fit(documents, weights)
        self.pickle_me()
        pass

    @staticmethod
    def process_tweets_for_babbler(raw_data):
        documents = [tweet["text"] for tweet in raw_data.itervalues()]
        weights = Agent.tweet_weights_from_raw_data(raw_data)
        return documents, weights

    def amnesia(self, hours_ago):
        # clear observational memory before hours_ago hours
        total_forgotten = 0
        for key in self.observational_memory.keys():
            tweet = self.observational_memory[key]
            current_hours_old = float((dt.datetime.now() - tweet["created_at"]).seconds) / 3600
            if current_hours_old > hours_ago:
                del self.observational_memory[key]
                total_forgotten += 1
        print "Forgot {} tweets.".format(total_forgotten)

    def oldest_tweet_id(self, topic):
        return np.min([int(tweet_id) for tweet_id in self.topic_memory[topic].iterkeys()])

    def newest_tweet_id(self, topic):
        if self.topic_memory[topic]:
            return np.max([int(tweet_id) for tweet_id in self.topic_memory[topic].iterkeys()])
        else:
            return 0


    def get_n_best_tweets(self, topic, n=5, metric="popularity", worst=False):
        return sorted(self.topic_memory[topic].items(), key=lambda val: val[1][metric], reverse=not worst)[:n]
        # weights = self.tweet_weights(topic)
        # sorted(weights)

    @staticmethod
    def tweet_weights_from_raw_data(topic_data):
        viralities = MinMaxScaler().fit_transform([tweet["virality"] for tweet in topic_data.itervalues()])
        popularities = MinMaxScaler().fit_transform([tweet["popularity"] for tweet in topic_data.itervalues()])
        weights = viralities + popularities
        return weights

    def tweet_weights(self, topic):
        return self.tweet_weights_from_raw_data(self.topic_memory[topic])

    def view_older_tweets(self):
        self.get_tweets_from_home_timeline(max_id=self.oldest_tweet_id)
        self.compress_and_store_tweets(self.tweet_test)
        pass

    def view_newer_tweets(self):
        self.get_tweets_from_home_timeline(since_id=self.newest_tweet_id)
        self.compress_and_store_tweets(self.tweet_test)
        pass

    def fit_tfidf(self):
        print "Fitting Tf-Idf"
        documents = [tweet["stemmed_text"] for tweet in self.observational_memory.itervalues()]
        self.tfidf = TfidfVectorizer(stop_words="english")
        self.transformed_documents = self.tfidf.fit_transform(documents)
        return self.tfidf, self.transformed_documents


    def apply_svd(self, n_components=10):
        print "Applying SVD"
        self.svd = TruncatedSVD(n_components=n_components)
        self.reduced_documents = self.svd.fit_transform(self.transformed_documents)


    def fit_linear_model(self):
        print "Fitting Linear Model"
        regr = LinearRegression()
        data = train_test_split(self.reduced_documents, self.tweet_weights)

        # regr.fit(features_train, features_test)
        regr, data, acc = fit_and_score(regr, data)
        self.regr = regr
        return regr, data, acc

    def fit_nocontent_linear_model(self, topic):
        print "Fitting linear model on high-level characteristics."
        regr = Lasso()
        feature_list = ["char_length", "total_words", "unique_words", "content_tag_ratio"]
        features = self.get_tweet_features(topic, feature_list)
        data = train_test_split(features, self.tweet_weights(topic))
        regr, data, acc = fit_and_score(regr, data)
        self.regr = regr
        print "Accuracy ", acc
        return regr, data, acc

    def fit_neural_network(self, topic):
        feature_list = ["char_length", "total_words", "unique_words", "content_tag_ratio"]
        features = MinMaxScaler().fit_transform(self.get_tweet_features(topic, feature_list))
        data = train_test_split(features, self.tweet_weights(topic), 0.20)
        features_train, features_test, labels_train, labels_test = data
        nn = Regressor(
            layers=[
                Layer("Rectifier", units=20),
                Layer("Rectifier", units=20),
                Layer("Linear")],
            learning_rate=0.02, n_iter=20,
            valid_set=(np.array(features_test), np.array(labels_test)))
        nn, data, acc = fit_and_score(nn, data)
        self.nn = nn
        return nn, data, acc

        rs = RandomizedSearchCV(nn, param_distributions={
            'learning_rate': stats.uniform(0.001, 0.05),
            'hidden0__units': stats.randint(5, 50),
            'hidden0__type': ["Rectifier", "Sigmoid", "Tanh"],
            'hidden1__units': stats.randint(5, 50),
            'hidden1__type': ["Rectifier", "Sigmoid", "Tanh"],
            'n_iter': stats.uniform(5,20),
        }, n_iter=10)
        rs, data, acc = fit_and_score(rs, data)
        self.nn = rs
        return rs, data, acc


    def learn_from_topic(self, topic):
        self.generate_babbler(topic)
        tfidf = self.fit_tfidf()
        svd = self.apply_svd()
        # regr, data, acc = self.fit_linear_model()
        nn, data, acc = self.fit_linear_model()

        # nn, data, acc = self.fit_neural_network()
        return dict(tfidf=tfidf, svd=svd, nn=nn, data=data, acc=acc)


    def tokenize_for_dedupe(self, tweet_text):
        clean_text = stem_and_tokenize(tweet_text)[3]
        # features = self.tfidf.transform([tweet_text])
        # features = self.svd.transform(features)
        # predicted_value = self.regr.predict(features)[0]
        # predicted_value = None
        return clean_text

    def get_tweet_features(self, topic, feature_list=[]):
        features = []
        for tweet in self.topic_memory[topic].itervalues():
            row = [tweet[feature] for feature in feature_list]
            features.append(row)
        return np.array(features)

    def babble(self, topic):
        if topic not in self.topic_memory:
            return None, None
        require = "#"+topic[3:]
        num_content, tweet_babble = self.topic_babblers[topic].babble(require="#"+topic[3:])
        return num_content, tweet_babble

    def babble_and_evaluate_one(self, topic):
        num_content, new_tweet = self.babble(topic)
        if not num_content:
            return None, None
        clean_text = self.tokenize_for_dedupe(new_tweet)
        print "Clean_text ", clean_text
        print num_content, "content words"
        for k,stemmed_text in [(k, tweet["stemmed_text"]) for k,tweet in self.topic_memory[topic].iteritems()]:
            if clean_text in stemmed_text or (stemmed_text in clean_text and len(stemmed_text.split())>3):
                print "DUPE!"
                current_tweet = self.topic_memory[topic][k]
                if "retweeted_already" not in current_tweet:
                    if random.random() < (current_tweet["popularity"] + current_tweet["virality"]) / 20:
                        print "RETWEETING!!!"
                        eric.retweet(k)
                        self.topic_memory[topic][k]["retweeted_already"] = True
                return None, None
        for stemmed_text in [tweet["stemmed_text"] for tweet in self.personal_memory.itervalues()]:
            if clean_text in stemmed_text or (stemmed_text in clean_text and len(stemmed_text.split()) > 3):
                print "DUPE!"
                return None, None

        return num_content, new_tweet

    def generate_tweet_pool(self, topic, n=40, stubborn=100):
        tweet_pool = []
        original_stubborn = stubborn
        while len(tweet_pool) < n and stubborn > 0:
            num_content, tweet = self.babble_and_evaluate_one(topic)
            if num_content:
                tweet_pool.append([num_content, tweet])
                stubborn = original_stubborn
            else:
                stubborn -= 1
            print "Found so far: ", len(tweet_pool)
        return sorted(tweet_pool, key=lambda row: row[0], reverse=True)

    def get_n_best_from_pool(self, topic, n=5, pool_size=100):
        tweet_pool = self.generate_tweet_pool(topic, n=pool_size)
        return tweet_pool[:n], tweet_pool

    def get_n_random_from_pool(self, topic, n=5, pool_size=100, lower_percentile=.2):
        tweet_pool = self.generate_tweet_pool(topic, n=pool_size)
        # cut off the lower chunk of shorter tweets
        tweet_pool = tweet_pool[:int(len(tweet_pool)*(1-lower_percentile))]
        total_content = np.sum([row[0] for row in tweet_pool])
        tweet_candidates = [row[1] for row in tweet_pool]
        length_weights = [float(row[0])/total_content for row in tweet_pool]
        if len(tweet_pool) > n * 2:
            return np.random.choice(tweet_candidates, n, p=length_weights), tweet_pool
        else:
            return None, tweet_pool


    def post_n_random_from_pool(self, topic, n=2, pool_size=100, lower_percentile=0.2):
        choices, pool = self.get_n_random_from_pool(topic=topic, n=n, pool_size=pool_size,
                                                    lower_percentile=lower_percentile)
        if choices:
            print "I have choices"
            tweets = set(choices)
            if len(pool) > n * 5:
                print "I'm gonna try to post."
                self.post_multiple_tweets(tweets)

    def post_multiple_tweets(self, tweets):
        print "Trying to post"
        for tweet_text in tweets:
            print "I have a tweet"
            try:
                print "Trying to post it."
                self.post_tweet(unicode(tweet_text))
            except TwitterHTTPError as ex:
                print "Fucking twitter", ex.message
                return

    def post_tweet(self, tweet_text):
        if not self.already_posted(tweet_text):
            print u"Posting '{}'".format(tweet_text)
            try:
                result = self.tw.statuses.update(status=tweet_text.encode("utf-8"))
                self.tweet_result = result
                self.compress_and_store_tweets([result], personal=True)
            except TwitterHTTPError as ex:
                warn("Fucking twitter: " + ex.message )

        else:
            print "Already posted a tweet just like that."

    def retweet(self, tweet_id):
        try:
            result = getattr(self.tw.statuses.retweet, tweet_id)(_id=tweet_id)
        except TwitterHTTPError as ex:
            warn("Fucking twitter: " + ex.message)

    def already_posted(self, tweet_text):
        stemmed_text = stem_and_tokenize(tweet_text)[3]
        if stemmed_text in [val["stemmed_text"] for val in self.personal_memory.itervalues()]:
            return True
        else:
            return False


    def get_possible_topics(self):
        try:
            trends = self.place_trends[0]["trends"]
            return {trend["query"]: trend["tweet_volume"] for trend in trends if trend["query"][:3]=="%23"}
        except IndexError:
            print "I don't know anything, I'm a fuckhead."


    def search_for(self, query, kwargs):
        try:
            return self.tw.search.tweets(q=query, **kwargs)
        except TwitterHTTPError as ex:
            warn(ex.message)
            time.sleep(30)
            return self.search_for(query, kwargs)

    def get_topic_tweets(self, topic, since_id=None, max_id=None, newer=True, **kwargs):
        default_args = dict(count=100, exclude_replies=False, lang="en")
        if since_id:
            default_args["since_id"] = since_id
        elif topic in self.topic_memory and newer:
            default_args["since_id"] = self.newest_tweet_id(topic)
        if max_id:
            default_args["max_id"] = max_id
        elif topic in self.topic_memory and not newer:
            default_args["max_id"] = self.oldest_tweet_id(topic)

        default_args.update(**kwargs)
        results = self.search_for(topic + " -RT", default_args)
        tweet_data = results["statuses"]
        if topic not in self.topic_memory:
            print "Looks like a new topic!"
            self.topic_memory[topic] = {}
        print "Storing results."
        added = self.compress_and_store_tweets(tweet_data, topic=topic)
        print "Looks like a success."
        return added




BRAIN_PICKLE = "erictau_pickle.p"
# brain = unpickle_me(BRAIN_PICKLE)
eric = Agent(tw=bot, pickle_file=BRAIN_PICKLE)

# agent.store_suggestion_groups()
# agent.store_all_suggested_users()
# agent.get_suggested_users("entertainment")

# agent.get_tweets_from_home_timeline()
# eric.pickle_me()
#
# eric.seed_followers_from_suggestions(refresh_groups=True)
#
# eric.pickle_me()
#

# print agent.suggestion_groups
# print len(agent.suggestion_groups[0])
# print agent.suggested_users
# print brain
#
# user = agent.suggested_users[0]["users"][0]
#
# for user in agent.suggested_users[0]["users"]:
#     print user["followers_count"], user["friends_count"]
#
# print agent.suggested_users[0]
# print agent.suggested_usergroups
# for ug in agent.suggested_usergroups: print len(ug)
#
# for key, val in user.items():
#     print key, val

#
# for tw in eric.tweet_test:
#     print tw["text"]
#
#
# print len(eric.universe)
# eric.pickle_me()
# eric.get_tweets_from_home_timeline()
# eric.compress_and_store_tweets(eric.tweet_test)
# print len(eric.observational_memory)
#
#
# vir_arr = np.array([item["virality"] for item in eric.observational_memory.itervalues()])
#
# print vir_arr
# print np.sum(vir_arr)
#
# pop_arr = np.array([item["popularity"] for item in eric.observational_memory.itervalues()])
# print pop_arr
#
# for tw in eric.observational_memory.itervalues():
#     print tw["favorites"]
#
# total = 0
# for k, tw in eric.observational_memory.items():
#     age = (dt.datetime.now() - tw["created_at"]).seconds / float(3600)
#     print age
#     if age>2.95:
#         total +=1
# print total
# eric.amnesia(3)
# eric.pickle_me()

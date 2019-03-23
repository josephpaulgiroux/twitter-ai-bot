

from MY import *

pp(eric.get_possible_topics())
print eric

eric.generate_master_babbler()

# topic = u"%23InspireSomeoneIn5WordsOrLess"
# eric.fit_nocontent_linear_model(topic)
# best, pool = eric.get_n_random_from_pool(topic)
# pp(best)

eric.pickle_me()

eric.master_babbler3.babble()

topics = eric.get_possible_topics()
for topic in topics:
    # if "Inspire" not in topic and (topic not in eric.topic_memory or len(eric.topic_memory[topic]) < 900):
    eric.get_topic_tweets(topic)
    for n in xrange(0,10):
        eric.get_topic_tweets(topic, newer=False)
    eric.generate_babbler(topic)

for n in xrange(0,3):
    for topic in topics:
        eric.post_n_random_from_pool(topic, n=1, pool_size=30)

eric.get_topic_tweets(topic)

eric.generate_babbler(topic)

# best, pool = eric.get_n_best_from_pool(topic)

eric.post_n_random_from_pool(topic, n=2)

pp(best)
pp(pool)
eric.fit_neural_network(topic)
eric.show_n_best_tweets(topic, 10)
eric.generate_babbler(topic)

eric.babble_and_evaluate_one(topic)


eric.get_topic_tweets(topic)
print eric

eric.pickle_me()
print eric
eric.generate_babbler(topic)
eric.babble(topic)

# eric.seed_followers_from_suggestions(refresh_groups=True)
eric.pickle_me()
eric.get_place_trends()


for tr in eric.place_trends[0]:
    print tr
    for tr in tr["trends"]:
        for k,v in tr.iteritems():
            print k, v

# eric.view_older_tweets()
# eric.view_newer_tweets()
print eric
eric.get_trend_locations()
eric.generate_babbler()

results = eric.learn_from_tweets()
eric.babbler.babble()

eric.babble_and_evaluate_one()

print results["acc"]


eric.babble_and_evaluate_one()
vec, features = eric.fit_tfidf()

# eric.amnesia(3)
from nltk.tokenize.casual import casual_tokenize

for k,v in eric.observational_memory.items():
    try:
        print "{}".format(unicode(v["text"]))
    except:
        print "***"
        print casual_tokenize(v["text"])

print unicode(u'\U0001f98b')



for k,v in eric.observational_memory.items():
    eric.observational_memory[k]["stemmed_text"] = stem_and_tokenize(v["text"])[3]
eric.pickle_me()

for item in vec.get_feature_names(): print item


from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=100)
reduced = svd.fit_transform(features)
print reduced
print features
for item in dir(svd): print item

print svd.explained_variance_ratio_

stim_first = [1] + [0] * 99
orig = svd.inverse_transform(stim_first)
highest = 0
for n, val in enumerate(list(orig[0])):
    if val > highest:
        highest = val
        print "Best so far is {}".format(vec.get_feature_names()[n])




stim_second = [0] + [1] + [0] * 98
orig = svd.inverse_transform(stim_second)
highest = 0
for n, val in enumerate(list(orig[0])):
    if val > highest:
        highest = val
        print "Best so far is {}".format(vec.get_feature_names()[n])


for tw in eric.observational_memory.values():
    if "Watching" in tw["text"]:
       print tw["text"]
       print stem_and_tokenize(tw["text"])


for tw in eric.observational_memory.values():
    if "ACLU" in tw["text"]:
       print tw["text"]
       print stem_and_tokenize(tw["text"])

for tw in eric.observational_memory.values():
    if "interesting" in tw["text"]:
       print tw["text"]
       print tw["stemmed_text"]
       print stem_and_tokenize(tw["text"])




for k, tw in eric.topic_memory[topic].items():
    eric.topic_memory[topic][k]["stemmed_text"] = stem_and_tokenize(tw["text"])[3]
eric.pickle_me()

for tw in eric.topic_memory[topic].values():
    if "least" in tw["text"]:
       print tw["text"]
       print tw["stemmed_text"]
       print stem_and_tokenize(tw["text"])



for tw in eric.observational_memory.values():
    if tw["user_id"] not in eric.universe:
        eric.follow_user(tw["user_id"])

for k, tw in eric.observational_memory.items():
    if not eric.observational_memory[k]["followers"]:
        print tw
    eric.observational_memory[k]["virality"] = (float(
        eric.observational_memory[k]["retweets"])/eric.observational_memory[k]["followers"]
                                                      )/(eric.observational_memory[k]["hours_old"]+1)
    eric.observational_memory[k]["popularity"] = (float(
        eric.observational_memory[k]["favorites"]) / eric.observational_memory[k]["followers"]
                                                      ) / (eric.observational_memory[k]["hours_old"]+1)




for k, tw in eric.observational_memory.items():
    print tw["popularity"], tw["virality"]

for k, tw in eric.topic_memory[topic].items():
    eric.topic_memory[topic][k]["popularity"] = float(tw["favorites"])/tw["followers"]
    eric.topic_memory[topic][k]["virality"] = float(tw["retweets"]) / tw["followers"]

eric.pickle_me()

for k, tw in eric.topic_memory["%23BadBreakupLocations"].items():
    if "alkie" in tw["text"]:
        print tw["text"]

import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

TRAIN = 'train'
TEST = 'test'

#fetch data from model
data = fetch_movielens(min_rating = 4.0)

#print training and testing data
print(repr(data[TRAIN]))
print(repr(data[TEST]))

#create model
model = LightFM(loss = 'warp')

#train mode
model.fit(data[TRAIN], epochs=30, num_threads=2)

#recommender fucntion
def sample_recommendation(model, data, user_ids):
    #number of users and movies in training data
    n_users, n_items = data[TRAIN].shape
    for user_id in user_ids:
    	#movies they already like
        known_positives = data['item_labels'][data[TRAIN].tocsr()[user_id].indices]

        #movies our model predicts they will like
        scores = model.predict(user_id, np.arange(n_items))
        #sort them in order of most liked to least
        top_items = data['item_labels'][np.argsort(-scores)]
        #print out the results
        print("User %s" % user_id)
        print("     Known positives:")

        for x in known_positives[:3]:
            print("        %s" % x)

        print("     Recommended:")

        for x in top_items[:3]:
            print("        %s" % x)
            
sample_recommendation(model, data, [3, 25, 451])






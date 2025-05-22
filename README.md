# fair-influence-embedding
This code is implementation of Fair2vec for the paper titled "Fair2Vec: Learning Fair and Topic-Aware
 Representations for Influencer Recommendation".

network.py is the multitask learning model. It is trained for each topic and the saved model for each topic is stored in the folder "saved model".

probability_of_influence.py file calculates the topic wise probality of influence between the influencer and follower.
it applies the sigmoid function on the dot product of influencer and follower embedding and calculates the probality of influence and store the topic wise probablity in the
file "dict_probability_of_influence".

seed_finder.py finds the k best influencers.
tag_finder.py finds the r best topics for the k best influencer.

Run `python tag_finder.py' to find top 5 influencial members and top 2 influence topics.

Run `python tag_finder.py number_of_influencial_users number_of_influence_tags' to get the output as top-k influencial nodes and top-r influence topics.

To train the model again -> Run `network.py'

To complete the entire process of training the model, finding probablity of influence and find the top fair influencers and top topics - Run python network.py && python probability_of_influence.py && python tag_finder.py

A dummy data is provided on which this code is implemented




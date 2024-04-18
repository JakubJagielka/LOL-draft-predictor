# LOL-draft-predictor
I want to create a model that will be able to predict the result of the League of Legends game after the draft phase.
I plan to use rich insights that I train in another model and then use them in the final model with transofrmers architecture.

#Current state
When training embeddings on 870 000 master+ games with context about the game and then transofrming them to train on no new model with no context about games, the model struggles to get good results.

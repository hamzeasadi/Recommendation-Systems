import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel




meta_data = pd.read_csv('./../data/movies_metadata.csv', low_memory=False)



tfidf = TfidfVectorizer(stop_words='english')
meta_data['overview'] = meta_data['overview'].fillna('')
tfidf_matrix = tfidf.fit_transform(meta_data['overview'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
print(cosine_sim.shape)
indices = pd.Series(meta_data.index, index=meta_data['title']).drop_duplicates()


def get_recommendations(title, cosine_sim = cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return meta_data['title'].iloc[movie_indices]


print(get_recommendations('The Godfather'))
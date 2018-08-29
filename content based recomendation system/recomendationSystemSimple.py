import pandas as pd

def weighted_rating(x, m=m, c=C):
    v = x['vote_count']
    R = x['vote_average']
    rw = (v/(v+m))*R + (m/(m+c))*R
    return rw


meta_data = pd.read_csv('movies_metadata.csv', low_memory=False)

#print(meta_data.head(3))
#print(meta_data['vote_count'])
#print(meta_data['vote_average'])
C = meta_data['vote_average'].mean()
#print(meta_data['vote_average'])
m = meta_data['vote_count'].quantile(0.90)
#x = meta_data['vote_count']
#print(len(x))
#print(m)
#print(C)
q_movies = meta_data.copy().loc[meta_data['vote_count']>=m]
q_movies['score'] = q_movies.apply(weighted_rating, axis=1)
q_movies = q_movies.sort_values('score', ascending=False)
print(q_movies[['title', 'vote_count', 'vote_average', 'score']].head(15))
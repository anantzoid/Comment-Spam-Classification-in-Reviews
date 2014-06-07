from sklearn.externals import joblib
import fextract1
from sklearn.feature_extraction.text import CountVectorizer

def loadmodels():
    rf = joblib.load('random_forest.pkl')
    svm = joblib.load('linearsvm.pkl')
    vect = joblib.load('vectorizer.pkl')

    return [svm,rf,vect]


if __name__ == "__main__":

    print "loading models..."
    svm,rf,vect = loadmodels()

    content = ['jncjks','Hi Anant, :D',"jhgfj","osum",'Reviews needed. Hurry up!']


    test_vector = vect.transform(content)

    test_features = fextract1.addfeatures(content, test_vector, yes = 1)
    print type(test_features)
    predictions = svm.predict(test_features)
    print predictions
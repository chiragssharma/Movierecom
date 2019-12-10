#Import system packages
import sys
#Import SparkContext and SparkConf libraries from pyspark 
from pyspark import SparkConf, SparkContext
#Import square root function from math library
from math import sqrt

#Define new functionn loadMovieNames which will return movieNames dictionary which will contain id and moviename
def loadMovieNames():
    movieNames = {}  #Define movieNames dictionary which will later have key,value pair of movie id & name
    with open("/home/cloudera/moviedata/itemfile.txt") as f: # Read itemfile & provide its characteristics to object f
        for line in f:  #read above itemfile which contains movie_id,movie name info line by line
            fields = line.split('|') #split the pipe delimited fields and store in a list variable fields
            movieNames[int(fields[0])] = fields[1].decode('ascii', 'ignore') # convert movie names in ASCII and assign it to corresponding movie id
    return movieNames #Return dictionary containing movie ids and movies names

# makePairs function will take parameters userID and ratings(i.e ((movie1,rating1),(movie2,rating2)))
#And will create pair of ((movie1,movie2),(rating1,rating2))
#Eg (01,((100,5),(200,5))
def makePairs((user, ratings)):
    (movie1, rating1) = ratings[0] #movie and rating is seperated as per eg.movie1=100,rating1=5
    (movie2, rating2) = ratings[1] #movie and rating is seperated. as per eg movie2=200,rating=5
    return ((movie1, movie2), (rating1, rating2)) #Return ((100,200),(5,5))
#This function filterDuplicates will remove duplicates created by movie and rating.
#For e.g Case 1 (01,(100,5),(200,5)) & Case 2 (01,(200,5),(100,5)) will return
# TRUE for Case 1. Since essentially Case 1 and Case 2 are the same so we need to
#consider only Case 1. Similarly duplicate pariring of same movies and same user
#would also be eliminated.
#Below para takes userID and ratings which is movie and rating pair.
#Eg (01,((100,5),(200,5))
def filterDuplicates( (userID, ratings) ):
    (movie1, rating1) = ratings[0] #movie and rating is seperated as per eg.movie1=100,rating1=5
    (movie2, rating2) = ratings[1] #movie and rating is seperated. as per eg movie2=200,rating=5
    return movie1 < movie2 #compare pairs and return TRUE if movie1<movie2. as per eg. return TRUE

#Function computeCosineSimilarity takes the rating pair for single movie and compute cosine similarity coefficient
#Formula for it is cos(theta) = a.b/|a|*|b|
def computeCosineSimilarity(ratingPairs):
    numPairs = 0 #initialize pair processed
    sum_xx = sum_yy = sum_xy = 0 #initialize
    for ratingX, ratingY in ratingPairs: #loop for each pair of ratings
        sum_xx += ratingX * ratingX #sum of multiplication of ratings of movie1
        sum_yy += ratingY * ratingY #sum of multiplication of ratings of movie2
        sum_xy += ratingX * ratingY #sum of multiplication of ratings of movie1 and movie2
        numPairs += 1 #count of no of pairs processed

    numerator = sum_xy 
    denominator = sqrt(sum_xx) * sqrt(sum_yy) #square root of sum_xx * square root of sum_yy

    score = 0 #initialize score to zero
    if (denominator):#check if denominator is non-zero
        score = (numerator / (float(denominator))) #score = numerator/denominator

    return (score, numPairs) #return score and count of no of pairs processed

#This is the telling application will run on local machine cluster with name of "MovieSimilarities"
#In short, SparkConf contains config information which is passed to SparkContext
conf = SparkConf().setMaster("local[*]").setAppName("MovieSimilarities")
#SparkContext is core of spark application. sc object is created to access spark and run my application
#MovieSimilarities on cluster.
sc = SparkContext(conf = conf)

print "\nLoading movie names..." #print
#As explained. This function will read u.item file and return dict having movie_id and movie name in nameDict
nameDict = loadMovieNames() 

data = sc.textFile("file:///home/cloudera/moviedata/datafile2.txt") #read ratings file on local system
#Rating file which is tab seperated. Hence split using split function. 
#Then using map function key value pair is created in format (key,(v1,v2)) 
#Here key = user-id, v1 =  movie_id, v2 = rating
ratings = data.map(lambda l: l.split()).map(lambda l: (int(l[0]), (int(l[1]), float(l[2]))))

#Self join ratings RDD with ratings RDD. This will join same user-id for all movie,rating pair
#So joinedRatings RDD will contain (user-id1,movie_id1,rating1),(movie_id2,rating2)),
#(user-id1,(movie_id2,rating2),(movie_id3,rating3)) and so on). Similarly for other userIDs.
joinedRatings = ratings.join(ratings)

#This will take joinedRatings RDD discussed above and keep only entries which return TRUE from function filterDuplicates
uniqueJoinedRatings = joinedRatings.filter(filterDuplicates)

#This will change the key to movie by movie pairings i.e(movie1,movie2)
moviePairs = uniqueJoinedRatings.map(makePairs)

#Join all the pairs which rate same group of movies.
#This will create a pair ((movie1,movie2),((rating1,rating2),(rating2,rating2)...etc
moviePairRatings = moviePairs.groupByKey()

#Now we will calculate similarity between movie pairs using the rating which is computed in func computeCosineSimilarity
#moviePairSimilarities will contain score and count of no of pairs processed for each movie pair combination
moviePairSimilarities = moviePairRatings.mapValues(computeCosineSimilarity).cache()

if (len(sys.argv) > 1): #verify if no of system arguments are more than 1 from where this pgm was executed. By default if we provide
#only program name no of argument is 1. So we need to pass movie_id in addition to pgm to execute below piece of code

    scoreThreshold = 0.10 #Score of similarity is set to .10
    coOccurenceThreshold = 2 #count of no pair of combination is equal to 2

    movieID = int(sys.argv[1]) #convert input movie_id for whom recommendation needs to be determined to INT
#Filter moviepair wherever movieID which was input from system matches AND
#score_of_similarity determined above is greater than .10
#count of no of movie pair combination is greater than 2
    filteredResults = moviePairSimilarities.filter(lambda((pair,sim)): \ 
        (pair[0] == movieID or pair[1] == movieID) \
        and sim[0] > scoreThreshold and sim[1] > coOccurenceThreshold)

#Sort movie pair in descending order of score and coOccurence. And take top 10. Also it flips the key value pair
    results = filteredResults.map(lambda((pair,sim)): (sim, pair)).sortByKey(ascending = False).take(10)


    print "Top 10 similar movies for " + nameDict[movieID] #Print movie name for which movie recommendation was needed
    for result in results: #Loop through results
        (sim, pair) = result #seperate out movie pair and score-coOccurence pair
        similarMovieID = pair[0] #search which movie of the pair is similar to one which was input
        if (similarMovieID == movieID):
            similarMovieID = pair[1]
        print nameDict[similarMovieID] + "\tscore: " + str(sim[0]) + "\tstrength: " + str(sim[1]) #print score and stength of 10 recommendations

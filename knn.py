# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from preprocessing.simplepreprocessor import SimplePreprocessor
from datasets.simpledatasetloader import SimpleDatasetLoader
from imutils import paths
import argparse
import time

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-k", "--neighbors", type=int, default=1,
help="# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1,
help="# of jobs for k-NN distance (-1 uses all available cores)")
ap.add_argument("-s", "--size", type=int, default=32,
help="size of the image preprocessed)")
args = vars(ap.parse_args())

# grab the list of images that we’ll be describing
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

# initialize the image preprocessor, load the dataset from disk,
# and reshape the data matrix
tick=time.clock()
sp = SimplePreprocessor(args["size"], args["size"])
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(imagePaths, verbose=1000)
data = data.reshape((data.shape[0], 3*args["size"]**2))

# show some information on memory consumption of the images
print("[INFO] features matrix: {:.1f}MB".format(data.nbytes / (1024 * 1000.0)))
tock=time.clock()
print("[INFO] Time on load and processing: {:.1f}s".format(tock-tick))

# encode the labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
test_size=0.25, random_state=42)

# train and evaluate a k-NN classifier on the raw pixel intensities
print("[INFO] evaluating k-NN classifier...")
model = KNeighborsClassifier(n_neighbors=args["neighbors"],
n_jobs=args["jobs"])
tick=time.clock()
model.fit(trainX, trainY)
tock=time.clock()
print("[INFO] Time on training: {:.1f}s".format(tock-tick))
tick=time.clock()
print(classification_report(testY, model.predict(testX),
target_names=le.classes_))
tock=time.clock()
print("[INFO] Time on evaluating: {:.1f}s".format(tock-tick))
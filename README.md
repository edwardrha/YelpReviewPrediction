# YelpReviewPrediction
Data science project for CSCI 183. ~~Using the data from Yelp reviews to predict the stars it will give.~~ Changed project to topic modeling and extraction of categorical scores for the restaurants based on the extracted topics.


# Steps to run our code:
Using: Python 2 and Jupyter Notebook
Required packages: numpy, pandas, sklearn, matplotlib, gensim, nltk, pytagcloud (optional tool for creating word clouds from https://github.com/atizo/PyTagCloud)

Setup:

1. Clone https://github.com/edwardrha/YelpReviewPrediction

2. Download Yelp data from https://www.yelp.com/dataset/challenge and place the JSON files into the /dataset directory.

3. Run the preprocessing.py from inside the /src directory. This will create and save the processed restaurant review files into the /dataset directory.
*WARNING:* This step requires very high amount of RAM(64GB or higher). We do not recommend this step to be run and instead use the processed JSON uploaded to Google Drive here: https://drive.google.com/drive/folders/1gYgWxNDK_78notWqKFuRiujawdkB640r?usp=sharing

4. Run the ClusterModel.py from inside the /src directory. This will create a CountVectorizer object, feature names object, train 6 LDA models for k=[10, 15, 20, 25, 30, 40] and pickle them into /models directory. It will also save the label predictions for the reviews into /dataset directory as txt files.
NOTE: Training each LDA model takes around 30 minutes each per core. Time consuming.

5. Now the Main.ipynb in the main directory is ready to be run.

*Main.ipynb:*
By using the models and data created from the Setup process, demonstrates how we can predict the category of a review and use it to give the categorical rating for a chosen restaurant.

*Labeling.ipynb:*
Contains the codes we used to examine the reviews from a cluster so we can manually label them.

*Slides.ipynb:*
Contains the codes and slides used to create our presentation. Run in terminal:
jupyter nbconvert Slides.ipynb --to slides --post serve
To open the presentation slides.

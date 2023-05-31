# Sentiment-analysis-using-text-in-the-image-information-but-not-normal-text
This project focuses on implementing sentiment analysis on text in image format, which is an emerging field in natural language processing. Sentiment analysis aims to determine the emotional tone or sentiment expressed in textual data. While sentiment analysis has traditionally been performed on text-based data like tweets, customer reviews, and news articles, the increasing use of images as a means of communication has made sentiment analysis on text in image format more important.

This repository explores various techniques and challenges involved in implementing sentiment analysis on text in image format. It provides a comprehensive overview of the data collection process, image pre-processing, text cleaning, model building, and evaluation of sentiment analysis for text in image format. The goal of this project is to help readers understand the complexities of analyzing sentiment from images and provide insights into sentiment analysis methods applied to text in image format.

# Business and People Insights
With the vast amount of textual and image data available, sentiment analysis has become a crucial tool for understanding customer feelings, product reviews, and social media trends. Analyzing sentiment in text in image format enables a deeper understanding of how people feel about a particular product or topic. However, sentiment analysis on text in image format presents unique challenges compared to traditional text-based sentiment analysis. For instance, images may contain multiple texts, text may be superimposed on complex backgrounds, or the font and text size may vary.

# Related Work
While sentiment analysis on textual data has been extensively studied, sentiment analysis on text in image format is a relatively new field with limited research. Here are some notable studies in this area:

Sharma et al. [1] proposed an approach that combines deep neural networks with standard machine learning methods for sentiment analysis of text embedded in images. Their approach demonstrated high accuracy and effectiveness in analyzing sentiment from text in image format.
Liu et al. [2] introduced a multi-model approach that combines visual characteristics extracted from the image with textual features from the embedded text for sentiment analysis on text in image format. Their approach outperformed using only textual features.
Chen et al. [3] utilized transfer learning and a pre-trained BERT model for sentiment analysis on text in image format. Their method achieved superior results compared to conventional techniques.
You et al. [4] proposed a deep convolutional neural network approach for sentiment analysis, achieving state-of-the-art results on datasets such as Flickr and Twitter.
These studies highlight the increasing demand and ongoing research in sentiment analysis of text in image format, leading to the development of more reliable and precise techniques.

# Data and Methods
The dataset used in this project consists of two movie reviews and two Amazon product reviews. These text datasets are utilized to train and test the sentiment analysis model.

The project follows the following steps:

Data Preprocessing: The data is preprocessed by tokenizing the text, removing stop words, and converting the reviews to equal length. Tokenization involves splitting the text into words and removing unnecessary characters or punctuation. Stop words, which do not contribute to sentiment, are eliminated. Each review is then standardized to a fixed length using padding or truncation.

Word Embeddings: Word embeddings are utilized to represent each word as a fixed-length vector. These vectors capture the semantic relationships and context of words in the input sequence. The word index file contains all distinct words and their corresponding indexes, allowing mapping of words in the reviews to their respective indexes.

Model Architecture: The model architecture consists of layers such as a word embedding layer, a global average pooling layer, dense layers with hidden units and ReLU activation functions, and an output layer with sigmoid activation function. This architecture helps the model learn non-linear relationships and predict sentiment based on input reviews.

# Results and Summary
1. Hair Straightener Electronic Gadget:

![straightner](https://github.com/pranav-nani/Sentiment-analysis-using-text-in-the-image-information-but-not-normal-text/assets/88759848/9610b763-5e61-49e0-be63-3d260ca22b77)

As we can see from the above fig our sentiment analysis model gave an accuracy of 84% means that our model correctly predicted the sentiment of 84% of the reviews in the test
dataset. This is a good indication of the model’s performance in terms of classification accuracy.

On the other hand, we got a binary cross-entropy loss function of 43% which means the model’s predicted probabilities for the correct class were, on average, 43% lower than the
actual probabilities. This measure shows how well the model is fitting the training data,with a lower value indicating a better fit. Overall, we got an 84% accuracy and 43% binary cross-entropy loss is a good performance for a sentiment analysis model
# Conclusion
In conclusion, we built a deep learning model that uses word embeddings and neural networks
to classify movie and product reviews by sentiment. We used four datasets containing reviews
and preprocessed the data by tokenizing, removing stop words, and equalizing the length of
every review. We then converted each word in the reviews into a vector of equal length using
word embeddings.

There are four layers in the model architecture: the embedding layer, the global average
pooling layer, the two dense layers, and the output layer. We train the model using Adam
optimization algorithms and binary cross-entropy losses. The model achieved 80% accuracy
on the validation set. To further improve the model’s performance, we can experiment with
changing the learning rate, the number of hidden layers, or the number of units in each layer.
Additionally, we can try using different pre-trained word embeddings or even train our own
embeddings on a larger dataset.
Overall, this project provides an excellent introduction to natural language processing and
deep learning techniques for sentiment analysis.

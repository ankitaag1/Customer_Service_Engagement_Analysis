# Install rtweet package to download data from X
install.packages("rtweet")
library(rtweet)

# Replace with your actual bearer token
auth <- rtweet_app(bearer_token = "YOUR_BEARER_TOKEN")

# Get tweets based on the search terms
tweets <- search_tweets(
  q = " " # Add actual search terms; Separate multiple terms with OR
  include_rts = FALSE,
  lang = "en"
)

# Save the tweets to a CSV file
write.csv(tweets, "tweets.csv", row.names = FALSE)

# Clean tweets
tweets$text <- gsub("http\\S+|www\\S+", "", tweets$text)  # Remove URLs
tweets$text <- gsub("@\\w+", "", tweets$text)  # Remove mentions
tweets$text <- gsub("#\\w+", "", tweets$text)  # Remove hashtags
tweets$text <- gsub("[[:punct:]]", "", tweets$text)  # Remove punctuation
tweets$text <- gsub("\\s+", " ", tweets$text)  # Remove extra spaces
tweets$text <- gsub("[0-9]+", "", tweets$text)  # Remove numbers
tweets$text <- gsub("[^[:alnum:][:space:]]", "", tweets$text)  # Remove special characters
tweets$text <- tolower(tweets$text) # Convert to lowercase

# Save the cleaned tweets to a new CSV file
write.csv(tweets, "cleaned_tweets.csv", row.names = FALSE)

# Install syuzhet package for sentiment analysis
install.packages("syuzhet")

# Load the syuzhet package 
library(syuzhet)
sentiment_scores <- get_sentiment(tweets$text, method = "syuzhet")
# Add sentiment scores to the tweets data frame
tweets$sentiment <- sentiment_scores

# Save the tweets with sentiment scores to a new CSV file
write.csv(tweets, "tweets_with_sentiment.csv", row.names = FALSE)

# Install packages for topic modeling
install.packages("tm")
install.packages("topicmodels")
install.packages("SnowballC")

# Load necessary libraries for topic modeling
library(tm)
library(topicmodels)
library(SnowballC)

# Create a corpus from the cleaned tweets
corpus <- VCorpus(VectorSource(tweets$text))
# Create a Document-Term Matrix
dtm <- DocumentTermMatrix(corpus)
# Fit the LDA model
lda_model <- LDA(dtm, k = 30, control = list(seed = 1234))
# Get the top 10 words for each topic
topics <- terms(lda_model, 10)

# Give dominant topic for each text
topic_assignments <- posterior(lda_model)$topics
dominant_topics <- apply(topic_assignments, 1, which.max)
# Display dominant topics for each text
for (i in seq_along(tweets$text)) {
  cat(sprintf("Text %d: Dominant Topic %d\n", i, dominant_topics[i]))
}
# Add dominant topic as column  
tweets$dominant_topic <- dominant_topics

# Install required packages to create a word cloud
install.packages("wordcloud")
install.packages("tm")
install.packages("RColorBrewer")

# Load libraries
library(wordcloud)
library(tm)
library(RColorBrewer)

# Create a text corpus
corpus <- VCorpus(VectorSource(tweets$text))

# Create a term-document matrix
tdm <- TermDocumentMatrix(corpus)
m <- as.matrix(tdm)
word_freqs <- sort(rowSums(m), decreasing = TRUE)
df <- data.frame(word = names(word_freqs), freq = word_freqs)
par(mar = c(1, 1, 1, 1))
# Generate word cloud with text
wordcloud(
  words = df$word,
  freq = df$freq,
  min.freq = 5,
  scale = c(4, 0.7),
  colors = rep("black", length(df$word)),
  random.order = FALSE
)

# After manually labeling each topic on the basis of words in each topic and five SERVQUAL (RATER) dimensions on the scale of 1-7, add these labels to each tweet text
# So, now each tweet text has a sentiment score as dependent variable and a score for each SERVQUAL dimensions namely X1, X2, X3, X4, X5 as independent variables 
# Perform multivariate regression analysis to predict sentiment based on SERVQUAL dimensions
model <- lm(sentiment ~ dimension1 + dimension2 + dimension3 + dimension4 + dimension5, data = tweets)
# Display the summary of the regression model
summary(model)
# Print coefficients for each independent variable
cat("\nCoefficients:\n")
print(coef(model))

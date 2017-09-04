setwd("C:/Users/Venkatramani/Desktop/Test")

# load the data
data <- read.csv('titanic_training.csv', header = TRUE)

# Exclude NA's
# check if row has NA's
row.has.na <- apply(data, 1, function(x){any(is.na(x))})
sum(row.has.na)
data.filtered <- data[!row.has.na,]

# check correlation between Age and Fare
c <- cor.test(data.filtered$Age,data.filtered$Fare,method = "pearson") 
s <- cor(data.filtered$Age,data.filtered$Fare,method = "spearman")
c
s
# we see 9.6% correlation between Age and the Fare. 

# plot(data.filtered$Age,data.filtered$Fare)


# Independece check
tb1 = table(data.filtered$Survived,data.filtered$Pclass)
tb1  # contingency table
# Setting `correct=FALSE` to turn off Yates' continuity correction.
chisq.test(tb1,correct = FALSE)

tb2 = table(data.filtered$Survived,data.filtered$Sex)
chisq.test(tb2)

tb3 = table(data.filtered$Survived,data.filtered$Age)
chisq.test(tb3)
# As the p-value is less than our significance level of 0.05 at 2.2e-16 
# (there is a really low probability that this occurred just by chance), 
# we fail to reject the null hypothesis. 

# Plotting distributions of Age vs Gender
plot(data.filtered$Age,data.filtered$Sex)




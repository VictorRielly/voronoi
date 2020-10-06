# my first kernel classifier
rm(list=ls())
set.seed(0.1)
library(readr) # CSV file I/O, e.g. the read_csv function
library(imager)
library(keras)
#library(reticulate)
#use_python('/usr/bin/python3')
# Specify what python to use. This was required for me
# to get keras to work properly. This may need to be 
# customized for each computer individually. You can findker
# the path by typing "which python" in the terminal in 
# linux (or mac) 
#######################################################
# We will use some pretrained models from keras for 
# imagenet to write some transfer learning code in R
#######################################################
# We drop the top layer because we are not interested in
# classifications, just features
model1<-keras::application_xception(weights='imagenet',include_top=FALSE)
model2<-keras::application_densenet201(weights='imagenet',include_top=FALSE)
# We need to define the image preprocessors
preprocessor1<-xception_preprocess_input
preprocessor2<-densenet_preprocess_input
#######################################################
# Some vectorized kernel functions, the inputs x, and y
# need to be data matrices
#######################################################
k1 = function(x,y)
{
	print(dim(x))
      print(dim(y))
	return(x%*%t(y))
}
k12 = function(x,y)
{
	return(x%*%t(y)+1)
}
k2 = function(x,y)
{
	return((x%*%t(y))^2)
}
k3 = function(x,y)
{
	return((1+x%*%t(y))^2)
}
d=4
k4 = function(x,y)
{
	return((1+x%*%t(y))^d)
}
sigma=1
########################################################
# This is the tricky kernel to vectorize
########################################################
k5 = function(x,y)
{
	x_times_y = -2*x%*%t(y)
	ysum = rowSums(y*y)
	xsum = rowSums(x*x)
	# Adds the vector x to each column of x_times_y and 
	# Adds the vector y to each row of the result
	xminusysquared = sweep(sweep(x_times_y,1,t(xsum),'+'),2,t(ysum),'+')
	return(exp(-xminusysquared/(2*sigma^2)))
}
kappa=1
theta=1
k6 = function(x,y)
	return (tanh(kappa*x%*%t(y)+theta))
# This kernel function implements the transfer learning
# kernel
k7 = function(x,y,model,preprocessor)
{
	xIm<-array_reshape(x,c(dim(x)[1],size,size,channels));
	yIm<-array_reshape(y,c(dim(x)[1],size,size,channels));
	xIm<-preprocessor(x);
  yIm<-preprocessor(y);
  print(dim(x));
  print(dim(xIm));
  print(dim(yIm));
  xFeatures<-model %>% predict(xIm);
  yFeatures<-model %>% predict(yIm);
  lengthXFeatures<-dim(xFeatures)[2]*dim(xFeatures)[3]*dim(xFeatures)[4];
  lengthYFeatures<-dim(yFeatures)[2]*dim(yFeatures)[3]*dim(yFeatures)[4];
	xFeatures<-array_reshape(xFeatures,c(dim(xFeatures)[1],lengthXFeatures));
  yFeatures<-array_reshape(yFeatures,c(dim(yFeatures)[1],lengthYFeatures));
  return(xFeatures%*%t(yFeatures));
}

k = function(x,y)
	return(k7(x,y,model1,preprocessor1))
##################################
## load the images 
##################################
train.path="C:\\Users\\19712\\Documents\\Bruno Python\\Cat versus dogs\\train\\train\\"
cats= list.files(path = train.path, pattern = "cat.+")
dogs= list.files(path = train.path, pattern = "dog.+")
size = 299
channels = 3
numTrain=100;
numTest=100;
train = c(cats[1:1000],dogs[1:1000])
train = sample(train,size=numTrain)
evaluation = c(cats[1001:1500],dogs[1001:1500])
evaluation = sample(evaluation,size=numTest)

train_prep = function(images, size, channels, path){
  count<- length(images)
  #master_array <- array(NA, dim=c(count,size, size, channels))
  master_array=c();
  for (i in seq(length(images))) {
    print(sprintf("loading image %d",i))
    ### load jpeg into imageR
    #img <- image_load(path = paste(path, images[i], sep=""), target_size = c(size,size))
    #img_arr <- image_to_array(img)
    img <- load.image(paste(path, images[i], sep=""))
    img <-  resize(img,size_x = size, size_y = size, size_c = channels)
    #img_arr <- array_reshape(img,c(size*size*channels))
    print(length(img))
    print(dim(img))
    #img_arr <- array_reshape(img, c(1, size, size, channels))
    #img_arr <- image_to_array(img)
    img_arr <- array_reshape(img, c(1, size, size, channels))
    #master_array[i,,,] <- img_arr
    master_array=cbind(master_array,c(img_arr))
    print(dim(master_array))
  }
  master_array = t(master_array)
  return(master_array)
}

x_train <- train_prep(train, size, channels, train.path)
x_evaluation <- train_prep(evaluation, size, channels,train.path)
y_train <- as.numeric(grepl("dog.", train, ignore.case = TRUE))
y_evaluation <- as.numeric(grepl("dog.", evaluation, ignore.case = TRUE))
##################################
## compute the classifier ########
##################################
n=length(y_train) # size of the training set
n.p=sum(y_train) # number of dogs
n.m=n-n.p #number of cats
#kk=outer(1:n,1:n,Vectorize(function(i,j) k(x_train[i,,,],x_train[j,,,])))
kk = k(x_train,x_train);
k.mm=kk[which(y_train==0),which(y_train==0)]
k.pp=kk[which(y_train==1),which(y_train==1)]
b=(sum(k.mm)/(n.m*n.m)-sum(k.pp)/(n.p*n.p))/2
alpha=ifelse(y_train,1/n.p,-1/n.m)
##################################
## empirical loss
##################################
y.hat = t(kk)%*%alpha+rep(b,n)
loss.emp=table(y_train,y.hat>0)
print(loss.emp)
print(sprintf("Empirical error rate: %.2f",(loss.emp[1,2]+loss.emp[2,1])/sum(loss.emp)))
##################################
## evaluate the classifier #######
##################################
#kk.t=outer(1:n,1:length(evaluation),Vectorize(function(ini,j) k(x_train[i,,,],x_evaluation[j,,,])))
kk.t=k(x_train,x_evaluation)
y.hat = t(kk.t)%*%alpha+rep(b,length(evaluation))
loss.hat=table(y_evaluation,y.hat>0)
print(loss.hat)
print(sprintf("Estimated error rate: %.2f",(loss.hat[1,2]+loss.hat[2,1])/sum(loss.hat)))
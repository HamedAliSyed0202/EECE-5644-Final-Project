%% ULTRASOUND 2-CLASS SVM (Benign vs Malignant)
% EECE 5644 – Final Project
% Syed Hamed Ali

clear; clc; close all;

%% STEP 1 — Load Dataset
rootDir = 'C:\Users\syedh\OneDrive\Desktop\FinalProjectEECE5644\archive\dataset_ultrasound';

imds = imageDatastore(rootDir, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

disp('Original dataset:');
countEachLabel(imds)

%% STEP 2 — Keep ONLY benign & malignant (remove normal)
imds = subset(imds, imds.Labels ~= 'normal');

disp('After removing normal class:');
countEachLabel(imds)

%% STEP 3 — Resize Images (for feature extraction)
imgSize = [224 224];
imds.ReadFcn = @(file) imresize(imread(file), imgSize);

%% STEP 4 — Train/Test Split (80/20)
[imdsTrain, imdsTest] = splitEachLabel(imds, 0.8, 'randomized');

%% STEP 5 — Extract HOG Features for Training
% Determine the length of a HOG feature vector
sampleImg = readimage(imdsTrain, 1);
[hogSample, ~] = extractHOGFeatures(sampleImg);
hogFeatureSize = length(hogSample);

% Pre-allocate feature matrix
trainFeatures = zeros(numel(imdsTrain.Files), hogFeatureSize);
trainLabels   = imdsTrain.Labels;

% Extract HOG for every training image
for i = 1:numel(imdsTrain.Files)
    img = readimage(imdsTrain, i);
    trainFeatures(i,:) = extractHOGFeatures(img);
end

%% STEP 6 — Train SVM Model (RBF Kernel)
SVMmodel = fitcsvm(trainFeatures, trainLabels, ...
    'KernelFunction', 'rbf', ...
    'KernelScale', 'auto', ...
    'Standardize', true);

%% STEP 7 — Extract HOG for Testing
testFeatures = zeros(numel(imdsTest.Files), hogFeatureSize);
testLabels   = imdsTest.Labels;

for i = 1:numel(imdsTest.Files)
    img = readimage(imdsTest, i);
    testFeatures(i,:) = extractHOGFeatures(img);
end

% Predict labels
predictedLabels = predict(SVMmodel, testFeatures);

%% STEP 8 — Evaluate Performance
accuracy = mean(predictedLabels == testLabels) * 100;
fprintf('\nUltrasound SVM Accuracy (2-Class): %.2f%%\n', accuracy);

% Remove empty category "normal" for a clean matrix
testLabels = removecats(testLabels);
predictedLabels = removecats(predictedLabels);

figure;
confusionchart(testLabels, predictedLabels);
title('Ultrasound SVM Classification Results (Benign vs Malignant)');

%% ULTRASOUND SVM CLASSIFICATION
% Syed Hamed Ali - EECE 5644 Final Project

clear; clc; close all;

%% STEP 1 — Load Dataset
rootDir = 'C:\Users\syedh\OneDrive\Desktop\FinalProjectEECE5644\archive\dataset_ultrasound';

imds = imageDatastore(rootDir, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

disp('Original dataset:');
countEachLabel(imds)


%% STEP 3 — Resize Images
imgSize = [224 224];
imds.ReadFcn = @(file) imresize(imread(file), imgSize);

%% STEP 4 — Train-Test Split
[imdsTrain, imdsTest] = splitEachLabel(imds, 0.8, 'randomized');

%% STEP 5 — Extract HOG Features for Training
exampleImg = readimage(imdsTrain, 1);
[hogFeat, ~] = extractHOGFeatures(exampleImg);
hogFeatureSize = length(hogFeat);

trainFeatures = zeros(numel(imdsTrain.Files), hogFeatureSize);
trainLabels   = imdsTrain.Labels;

for i = 1:numel(imdsTrain.Files)
    img = readimage(imdsTrain, i);
    trainFeatures(i,:) = extractHOGFeatures(img);
end

%% STEP 6 — Train SVM Model
SVMmodel = fitcsvm(trainFeatures, trainLabels, ...
    'KernelFunction', 'rbf', ...
    'KernelScale', 'auto', ...
    'Standardize', true);

%% STEP 7 — Extract HOG Features for Testing
testFeatures = zeros(numel(imdsTest.Files), hogFeatureSize);
testLabels   = imdsTest.Labels;

for i = 1:numel(imdsTest.Files)
    img = readimage(imdsTest, i);
    testFeatures(i,:) = extractHOGFeatures(img);
end

predictedLabels = predict(SVMmodel, testFeatures);

%% STEP 8 — Evaluate Performance
accuracy = mean(predictedLabels == testLabels) * 100;
fprintf('\nUltrasound SVM Accuracy: %.2f%%\n', accuracy);

figure;
confusionchart(testLabels, predictedLabels);
title('Ultrasound SVM Classification Results');

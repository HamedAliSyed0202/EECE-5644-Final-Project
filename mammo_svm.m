%% Mammography SVM Classification (Benign vs Malignant)

clear; clc; close all;

%% ======= PATH SETTINGS =======
baseDir = "C:\Users\syedh\OneDrive\Desktop\FinalProjectEECE5644\MammoGraphy\manifest-ZkhPvrLo5216730872708713142";

% The folder created earlier by build_mammo_dataset.m
dataDir = fullfile(baseDir, "mammo_dataset");

if ~isfolder(dataDir)
    error("Dataset folder NOT found: %s", dataDir);
end
fprintf("Using dataset folder: %s\n", dataDir);

%% ======= LOAD DATA =======
imds = imageDatastore(dataDir, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames', ...
    'FileExtensions', {'.png', '.jpg', '.jpeg'});

disp("Dataset loaded:");
countEachLabel(imds)

%% ======= TRAIN/TEST SPLIT =======
[imdsTrain, imdsTest] = splitEachLabel(imds, 0.8, 'randomized');

%% ======= FEATURE EXTRACTION (HOG) =======
fprintf("Extracting HOG features...\n");

hogFeatureSize = [];
trainingFeatures = [];
trainingLabels = imdsTrain.Labels;

for i = 1:numel(imdsTrain.Files)
    img = readimage(imdsTrain, i);
    img = imresize(img, [256 256]);
    if size(img,3)==3
        img = rgb2gray(img);
    end
    [feat, v] = extractHOGFeatures(img);

    if isempty(hogFeatureSize)
        hogFeatureSize = length(feat);
    end

    trainingFeatures(i,1:hogFeatureSize) = feat;
end

fprintf("HOG feature extraction done.\n");

%% ======= TRAIN SVM =======
classifier = fitcsvm(trainingFeatures, trainingLabels, ...
    'KernelFunction', 'rbf', ...
    'KernelScale', 'auto', ...
    'Standardize', true);

fprintf("SVM training completed.\n");

%% ======= TESTING =======
testFeatures = [];
testLabels = imdsTest.Labels;

for i = 1:numel(imdsTest.Files)
    img = readimage(imdsTest, i);
    img = imresize(img, [256 256]);
    if size(img,3)==3
        img = rgb2gray(img);
    end
    feat = extractHOGFeatures(img);
    testFeatures(i,1:hogFeatureSize) = feat;
end

predLabels = predict(classifier, testFeatures);

%% ======= ACCURACY =======
accuracy = mean(predLabels == testLabels) * 100;
fprintf("Mammography SVM Accuracy: %.2f%%\n", accuracy);

%% ======= CONFUSION MATRIX =======
figure;
confusionchart(testLabels, predLabels);
title("Mammography SVM Confusion Matrix");


%% Mammography CNN Classification (Custom CNN)
clc; clear; close all;

% -------------------------------------------------------
% PATH TO PROCESSED DATA
% -------------------------------------------------------
dataDir = "C:\Users\syedh\OneDrive\Desktop\FinalProjectEECE5644\MammoGraphy\manifest-ZkhPvrLo5216730872708713142\mammo_dataset";

fprintf("Using dataset: %s\n", dataDir);

% Load dataset
imds = imageDatastore(dataDir, ...
    "IncludeSubfolders", true, ...
    "LabelSource", "foldernames");

countEachLabel(imds)

% -------------------------------------------------------
% TRAIN/TEST SPLIT
% -------------------------------------------------------
[imdsTrain, imdsTest] = splitEachLabel(imds, 0.8, "randomized");

% Resize layer input size
inputSize = [224 224];  

augTrain = augmentedImageDatastore(inputSize, imdsTrain);
augTest  = augmentedImageDatastore(inputSize, imdsTest);

% -------------------------------------------------------
% CUSTOM CNN ARCHITECTURE
% -------------------------------------------------------
layers = [
    imageInputLayer([224 224 1],"Name","input")

    convolution2dLayer(5, 16, "Padding", "same")
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2, "Stride", 2)

    convolution2dLayer(3, 32, "Padding", "same")
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2, "Stride", 2)

    fullyConnectedLayer(64)
    reluLayer
    dropoutLayer(0.3)

    fullyConnectedLayer(2) 
    softmaxLayer
    classificationLayer
];

% -------------------------------------------------------
% TRAINING OPTIONS
% -------------------------------------------------------
options = trainingOptions("adam", ...
    "MaxEpochs", 12, ...
    "InitialLearnRate", 1e-4, ...
    "MiniBatchSize", 32, ...
    "Plots", "training-progress", ...
    "Shuffle", "every-epoch", ...
    "Verbose", true);

% -------------------------------------------------------
% TRAIN CNN
% -------------------------------------------------------
net = trainNetwork(augTrain, layers, options);

% -------------------------------------------------------
% TESTING
% -------------------------------------------------------
YPred = classify(net, augTest);
YTrue = imdsTest.Labels;

accuracy = mean(YPred == YTrue) * 100;
fprintf("\nMammography Custom CNN Accuracy: %.2f%%\n", accuracy);

% -------------------------------------------------------
% CONFUSION MATRIX
% -------------------------------------------------------
figure;
confusionchart(YTrue, YPred);
title("Mammography Custom CNN Confusion Matrix");

%% TRANSFER LEARNING ON MAMMOGRAPHY (ResNet18 + EfficientNetB0)
clc; clear; close all;

%% ======================
%  Load dataset
% =======================
dataDir = "C:\Users\syedh\OneDrive\Desktop\FinalProjectEECE5644\MammoGraphy\manifest-ZkhPvrLo5216730872708713142\mammo_dataset";

imds = imageDatastore(dataDir, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

labelCount = countEachLabel(imds)

% Train / test split
[imdsTrain, imdsTest] = splitEachLabel(imds, 0.8, "randomized");

%% Convert grayscale → RGB for pre-trained networks
inputSize = [224 224];
augmenter = imageDataAugmenter( ...
    'RandXReflection', true);

augTrain = augmentedImageDatastore(inputSize, imdsTrain, ...
    'ColorPreprocessing', 'gray2rgb', ...
    'DataAugmentation', augmenter);

augTest = augmentedImageDatastore(inputSize, imdsTest, ...
    'ColorPreprocessing', 'gray2rgb');

%% ==========================
%  TRAIN RESNET-18
% ==========================
disp(" ");
disp("=============================");
disp(" Training ResNet-18...");
disp("=============================");

net = resnet18;
lgraph = layerGraph(net);

numClasses = numel(categories(imdsTrain.Labels));

% Replace last FC layer (fc1000)
newFC = fullyConnectedLayer(numClasses, ...
    'Name', 'fc_custom', ...
    'WeightLearnRateFactor', 10, ...
    'BiasLearnRateFactor', 10);

lgraph = replaceLayer(lgraph, "fc1000", newFC);

% Replace Softmax (prob)
newSoftmax = softmaxLayer('Name','softmax_custom');
lgraph = replaceLayer(lgraph, "prob", newSoftmax);

% Replace Classification Layer
newClass = classificationLayer('Name','classoutput_custom');
lgraph = replaceLayer(lgraph, "ClassificationLayer_predictions", newClass);

% Training options
options = trainingOptions('adam', ...
    'MiniBatchSize', 32, ...
    'MaxEpochs', 6, ...
    'InitialLearnRate', 1e-4, ...
    'ValidationData', augTest, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'Plots', 'training-progress');

% Train ResNet18
resnetModel = trainNetwork(augTrain, lgraph, options);

% Evaluate
preds = classify(resnetModel, augTest);
resnetAccuracy = mean(preds == imdsTest.Labels) * 100;

fprintf("\n⭐ ResNet-18 Accuracy: %.2f%%\n", resnetAccuracy);


%% ==========================
%  TRAIN EfficientNet-B0
% ==========================
%% EfficientNet-B0 Transfer Learning (Fixed for Your MATLAB Version)

fprintf("\n=============================\n Training EfficientNet-B0...\n=============================\n");

net2 = efficientnetb0;
lgraph2 = layerGraph(net2);

numClasses = 2;

%% ---- Replace Final Layers ----

% Replace FC layer
newFC = fullyConnectedLayer(numClasses, "Name", "new_fc");
lgraph2 = replaceLayer(lgraph2, "efficientnet-b0|model|head|dense|MatMul", newFC);

% Replace Softmax
newSoft = softmaxLayer("Name", "new_softmax");
lgraph2 = replaceLayer(lgraph2, "Softmax", newSoft);

% Replace Classification Layer
newClass = classificationLayer("Name", "new_classoutput");
lgraph2 = replaceLayer(lgraph2, "classification", newClass);

%% ---- Image Augmentation ----
inputSize = net2.Layers(1).InputSize;

augTrain2 = augmentedImageDatastore(inputSize, imdsTrain, ...
    "ColorPreprocessing", "gray2rgb");

augTest2 = augmentedImageDatastore(inputSize, imdsTest, ...
    "ColorPreprocessing", "gray2rgb");

%% ---- Training Options ----
options2 = trainingOptions("adam", ...
    "MaxEpochs", 6, ...
    "InitialLearnRate", 1e-4, ...
    "MiniBatchSize", 16, ...
    "ValidationData", augTest2, ...
    "ValidationFrequency", 50, ...
    "Shuffle", "every-epoch", ...
    "Plots", "training-progress", ...
    "Verbose", true);

%% ---- Train EfficientNet-B0 ----
effnetModel = trainNetwork(augTrain2, lgraph2, options2);

%% ---- Evaluate ----
YPred2 = classify(effnetModel, augTest2);
YTest2 = imdsTest.Labels;

acc2 = mean(YPred2 == YTest2);
fprintf("\n⭐ EfficientNet-B0 Accuracy: %.2f%%\n", acc2*100);

figure;
confusionchart(YTest2, YPred2, "Title", "Mammography EfficientNet-B0 Confusion Matrix");

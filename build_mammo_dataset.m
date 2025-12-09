function build_mammo_dataset()

    % === PATHS ===
    baseDir = "C:\Users\syedh\OneDrive\Desktop\FinalProjectEECE5644\MammoGraphy\manifest-ZkhPvrLo5216730872708713142";
    cbisDir = fullfile(baseDir, "CBIS-DDSM");
    
    % Metadata CSV files (converted to CSV earlier)
    trainCSV = readtable(fullfile(baseDir, "mass_case_description_train_set.csv"));
    testCSV  = readtable(fullfile(baseDir, "mass_case_description_test_set.csv"));
    
    allData = [trainCSV; testCSV];
    
    % Output dataset folder
    outDir = fullfile(baseDir, "mammo_dataset");
    benignDir = fullfile(outDir, "benign");
    malignantDir = fullfile(outDir, "malignant");
    
    mkdir(outDir);
    mkdir(benignDir);
    mkdir(malignantDir);

    fprintf("Loaded metadata: %d entries.\n", height(allData));

    processedCount = 0;

    % Loop through metadata rows
    for i = 1:height(allData)

        pid = string(allData.patient_id(i));  % Example: P_00001
        label = lower(string(allData.pathology(i))); % benign / malignant
        
        % Find matching folder recursively
        patientFolder = find_patient_folder(cbisDir, pid);

        if patientFolder == ""
            fprintf("❌ Folder not found for: %s\n", pid);
            continue;
        end

        % Find .dcm inside the deepest folder
        dcmFile = find_dicom(patientFolder);

        if dcmFile == ""
            fprintf("❌ No DICOM image inside: %s\n", pid);
            continue;
        end

        % Read DICOM
        try
            img = dicomread(dcmFile);
            img = mat2gray(img); % Normalize
        catch
            fprintf("❌ Could not read DICOM for: %s\n", pid);
            continue;
        end

        % Save PNG
        if contains(label, "benign")
            outPath = fullfile(benignDir, pid + ".png");
        else
            outPath = fullfile(malignantDir, pid + ".png");
        end
        
        imwrite(img, outPath);

        processedCount = processedCount + 1;
        fprintf("✔ Saved: %s → %s\n", pid, label);
    end

    fprintf("\nDataset creation complete! %d images processed.\n", processedCount);
end


% === FIND PATIENT FOLDER RECURSIVELY ===
function folderPath = find_patient_folder(rootDir, pid)
    folders = dir(fullfile(rootDir, "**", "*" + pid + "*"));
    folderPath = "";

    for k = 1:length(folders)
        if folders(k).isdir
            folderPath = fullfile(folders(k).folder, folders(k).name);
            return;
        end
    end
end

% === FIND DICOM FILE INSIDE MULTI-LEVEL FOLDER ===
function dcmPath = find_dicom(patientFolder)
    files = dir(fullfile(patientFolder, "**", "*.dcm")); % search all levels
    if isempty(files)
        dcmPath = "";
    else
        dcmPath = fullfile(files(1).folder, files(1).name); % Take the first DICOM
    end
end

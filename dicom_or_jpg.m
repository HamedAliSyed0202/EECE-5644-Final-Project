function img = dicom_or_jpg(path)
    [~,~,ext] = fileparts(path);

    try
        if strcmpi(ext,".dcm")
            img = dicomread(path);
        else
            img = imread(path);
        end
    catch
        img = zeros(224,224,'uint8');
    end

    if size(img,3) == 3
        img = rgb2gray(img);
    end
end

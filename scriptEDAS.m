function [ result ] = scriptEDAS( cromosoma_ini )
%scriptEDAS Script para hacer uso de EDAS recibiendo cromosoma (array de
%len 5) donde cada gen es el valor (1-6) de la potencia de cada beacon


%%% Modificacion de script3.m
cromosoma = cromosoma_ini + 1;

%% Importa datos para el entrenamiento , de capturas previamente hechas
% Read the csv file
K = 5;
limit = 1;
trainingLabels= [];
trainingRssi = [];
validationLabel= [];
validationRssi = [];
matrixCSV = [];
matrixCSV2 = [];
matrixCSV3 = [];
nroPartitions = 23;
selectedBeacons = [1 2 3  4 5];

%% preparando archivos de entrenamiento
prefix = strcat('/home/jeslev/Dropbox/Bluetooth/ResultadosJesus/nuevosManuelBeacons/Raspberri/Tx_0x0',int2str(cromosoma(1)),'/train');
prefix2 = strcat('/home/jeslev/Dropbox/Bluetooth/ResultadosJesus/nuevosManuelBeacons/Raspberri/Tx_0x0',int2str(cromosoma(2)),'/train');
prefix3 = strcat('/home/jeslev/Dropbox/Bluetooth/ResultadosJesus/nuevosManuelBeacons/Raspberri/Tx_0x0',int2str(cromosoma(3)),'/train');
prefix4 = strcat('/home/jeslev/Dropbox/Bluetooth/ResultadosJesus/nuevosManuelBeacons/Raspberri/Tx_0x0',int2str(cromosoma(4)),'/train');
prefix5 = strcat('/home/jeslev/Dropbox/Bluetooth/ResultadosJesus/nuevosManuelBeacons/Raspberri/Tx_0x0',int2str(cromosoma(5)),'/train');


for it=1:15
    
    matrix1 = []; matrix2 = []; matrix3 = []; matrix4 = []; matrix5 = [];
    filename = strcat(prefix,int2str(it),'.csv');
    matrix1 = csvread(filename,0,0);
    filename = strcat(prefix2,int2str(it),'.csv');
    matrix2 = csvread(filename,0,0);
    filename = strcat(prefix3,int2str(it),'.csv');
    matrix3 = csvread(filename,0,0);
    filename = strcat(prefix4,int2str(it),'.csv');
    matrix4 = csvread(filename,0,0);
    filename = strcat(prefix5,int2str(it),'.csv');
    matrix5 = csvread(filename,0,0);
    
    % uniformiza tamaños
    tam = min([size(matrix1,1),size(matrix2,1),size(matrix3,1),...
                    size(matrix4,1),size(matrix5,1)]);
    matrixCSV = [matrix1(1:tam,1)';matrix1(1:tam,2)'; matrix2(1:tam,3)';
        matrix3(1:tam,4)';
        matrix4(1:tam,5)';matrix5(1:tam,6)']';
    
    trainingLabels = [trainingLabels; matrixCSV(:,1)];
    trainingRssi = [trainingRssi; matrixCSV(:,2:end)];
end
clear tmpLabels tmpRssi matrixCSV

%% Procesa archivos de validacion
prefix = strcat('/home/jeslev/Dropbox/Bluetooth/ResultadosJesus/nuevosManuelBeacons/Raspberri/Tx_0x0',int2str(cromosoma(1)),'/valid');
prefix2 = strcat('/home/jeslev/Dropbox/Bluetooth/ResultadosJesus/nuevosManuelBeacons/Raspberri/Tx_0x0',int2str(cromosoma(2)),'/valid');
prefix3 = strcat('/home/jeslev/Dropbox/Bluetooth/ResultadosJesus/nuevosManuelBeacons/Raspberri/Tx_0x0',int2str(cromosoma(3)),'/valid');
prefix4 = strcat('/home/jeslev/Dropbox/Bluetooth/ResultadosJesus/nuevosManuelBeacons/Raspberri/Tx_0x0',int2str(cromosoma(4)),'/valid');
prefix5 = strcat('/home/jeslev/Dropbox/Bluetooth/ResultadosJesus/nuevosManuelBeacons/Raspberri/Tx_0x0',int2str(cromosoma(5)),'/valid');

for it=1:15
    
    matrix1 = []; matrix2 = []; matrix3 = []; matrix4 = []; matrix5 = [];
    filename = strcat(prefix,int2str(it),'.csv');
    matrix1 = csvread(filename,0,0);
    filename = strcat(prefix2,int2str(it),'.csv');
    matrix2 = csvread(filename,0,0);
    filename = strcat(prefix3,int2str(it),'.csv');
    matrix3 = csvread(filename,0,0);
    filename = strcat(prefix4,int2str(it),'.csv');
    matrix4 = csvread(filename,0,0);
    filename = strcat(prefix5,int2str(it),'.csv');
    matrix5 = csvread(filename,0,0);
    
    tam = min([size(matrix1,1),size(matrix2,1),size(matrix3,1),...
                    size(matrix4,1),size(matrix5,1)]);
    matrixCSV = [matrix1(1:tam,1)';matrix1(1:tam,2)'; matrix2(1:tam,3)';
    matrix3(1:tam,4)'; matrix4(1:tam,5)';matrix5(1:tam,6)']';

    validationLabel = [validationLabel; matrixCSV(:,1)];
    validationRssi = [validationRssi; matrixCSV(:,2:end)];
end

%% Inicia algoritmo - KNN Search
[completeErrorMeasure, completeFinalResultsDistance, confussionMatrix, averageDistance,...
    correctPosFreq, totalPosFreq, confussion] = ...
    KNNprocess( trainingRssi, validationRssi,trainingLabels, validationLabel, K);

result = mean(correctPosFreq ./ totalPosFreq) * 100;
%% Final results after running 'reps' times the simulation process
%tabValues = tabulate(completeFinalResultsDistance);

%disp(completeErrorMeasure);
%disp(mean(completeErrorMeasure));

end


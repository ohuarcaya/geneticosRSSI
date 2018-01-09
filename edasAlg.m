% Alg EDAS para búsqueda de configuración correcta de beacons

nro_iter = 10;
times = zeros(1,nro_iter);

for it=1:nro_iter
    tic
    PopSize = 100; 
    n = 5; 
    cache  = [0,0,0,0,0]; 
    Card = 6*ones(1,n); 
    edaparams = {};
    F = 'scriptEDAS'; % objective function;
    [AllStat,Cache]=RunEDA(PopSize,n,F,Card,cache,edaparams) 
    save(strcat('results',int2str(it)),'AllStat');
    times(it) = toc;
end

save('tiempos','times')


return


%% mostrar resultados
load('statsEdas1.mat')
results = [];
for it=1:50
    config = AllStat(it,2);
    config_real = config{1}+1;
    results = [results;config_real];
end

disp(results)
#! /bin/bash

####################################################################################################

#necesita tener instalado perf y mongodb
#input: nada
# output: joules, intervalo de tiempo de muestra, ciclos, instrucciones, instrucciones/ciclo, %cpu,
# %memoria


##############################################################################################

echo "Calculando datos..."


sudo sh -c 'echo 0 > /proc/sys/kernel/perf_event_paranoid' 

#lanzamos los mensajes

#sudo ./mensajero.sh &

#perf stat -e power/energy-cores/ -ag sleep 5 2>&1 | sed '/^$/ d' | sed '1d' | sed 's/^ *//g' | tee-a salidaperfjoules.txt


#********************************************************************************************

#aqui cambie a background 5:06  8 marzo

variable=$(sudo perf stat -e power/energy-cores/ -ag sleep 1 2>&1 )
variable2=$(sudo perf stat -e cycles,instructions -ag sleep 1 2>&1 )
#variable=$(sudo perf stat test_modelo.py -I 1 sleep 10 2>&1 )

echo "$variable"

#**************************************************************************************

#perf record (guardamos los datos para verlos mas tarde)
#esta orden esta dentro del script record.sh

#sudo ./grabadorperf.sh &

sudo perf record -a sleep 1

#######################################
#
#---aqui un wait para esperar un proceso en background (aun no se cual espera)


#porcentajecpu=$(sudo perf report --sort comm --stdio | awk '{print $0} ' | grep mosquitto_pub | cut -d '%' -f1 | sed 's/ //g')

echo "****porcentaje de cpu = $porcentajecpu"



echo "$variable"
echo "$variable2"




tciclos=$(echo "${variable2:55:11}")

tinstruc=$(echo "${variable2:151:11}")
tinstxciclo=$(echo "${variable2:198:4}")

#instrucciones final
instrucciones=$(echo "$tinstruc" | sed -e 's/\.//g')
#inst/ciclo
instxciclo=$(echo "$tinstxciclo" | sed -e 's/\,/\./g')


echo "instrucciones/ciclo = $instxciclo"
echo "instrucciones = $instrucciones"

#ciclos final
ciclos=$(echo "$tciclos" | sed -e 's/\.//g')

echo "ciclos = $ciclos"

#echo $variable > salidaperfjoules.txt
energia=$(echo "$variable" | cut -d ":" -f 2 | cut -d "J" -f 1)

#energia final
energia2=$(echo ${energia:17:4})

#tiempo final
tiempo=$(echo ${energia:31:11})

echo "energia = $energia2";
echo "tiempo = $tiempo";










#joules, intervalo de tiempo de muestra, ciclos, instrucciones, instrucciones/ciclo

echo "$tiempo $energia2 $ciclos $instrucciones $instxciclo $porcentajecpu" | tee -a salidaperfjoules.txt

echo "$tiempo;$energia2;$ciclos;$instrucciones;$instxciclo;$porcentajecpu;" | tee -a salidaperfjoules.csv

#var2=$(perf stat -e cycles  sleep 5)

#echo "$variable"

#echo $variable;


#prueba tipos de variables
#var3=$(($energia2 + 2.0)) | bc -l

#echo "$energia2+2" | bc -l


#---------------------------------
#calculo de % cpu de proceso mosquitto_pub (publicacion de mensaje)

#porcentajecpu=$(sudo perf report --sort comm --stdio | awk '{print $0} ' | grep mosquitto_pub | cut -d '%' -f1)

#echo "-----------porcentaje de cpu = $porcentajecpu"

#>>>>>>>>ya lo esta haciendo grabadorperf.sh

#---------------------------------



#rutina de acceso a base de datos a mongodb


#mongo << EOF
#use benchmarkperf
#db.micoleccion.insert({tiempo:'$tiempo',energia:'$energia2',ciclos:'$ciclos',instrucciones:'$instrucciones',instxciclo:'$instxciclo'})
#quit()
#EOF

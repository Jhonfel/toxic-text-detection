#!/bin/bash

echo "Construyendo la imagen de Docker..."
docker build -t toxic-text-detection .

echo
echo "Ejecutando el contenedor..."
docker run -p 7000:7000 toxic-text-detection

echo
echo "El contenedor está corriendo. Puedes acceder a la aplicación en http://localhost:7000"
echo "Presiona Ctrl+C para detener el contenedor."
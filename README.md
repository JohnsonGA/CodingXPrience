![image](https://github.com/user-attachments/assets/94858d05-885f-4193-baec-2b38c98293db)# Taller_2_MLOps
## Integrantes:
- Luz Angela Iriarte Zamora
- Juan David Aristizabal Cifuentes
- Johnson Gómez Álvarez

## Dentro de esta instancia se encuentran los archivos:

- main.py
- requirements.txt
- MLOps_Trabajo_2_Heart_Attack_Analysis_&_Prediction_Dataset.ipynb
- logistic_pipeline.joblib
# Haciendo uso de replit y FastAPI 
Swagger FastAPI de predicciones del Joblib
![image](https://github.com/user-attachments/assets/c91c830a-32f0-4ec5-ab22-b8cca3569bd8)

![image](https://github.com/user-attachments/assets/ceb755e8-a069-4eb9-b32e-8bb19de43121)

# Desplegamosen EC2
Para el despliegue en EC2 necesitamos abrir la instancia e introducir las siguientes lineas

git clone https://github.com/JohnsonGA/CodingXPrience.git

sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update

sudo apt install python3.10
sudo apt install python3.10-venv python3-pip

# Crear y activar el entorno virtual
python3.10 -m venv mi_entorno
source mi_entorno/bin/activate

pip install -r requirements.txt

#Corremos el servicio temporalmente
python3 service.py

#Luego de gestionar la instancia de EC2 para tener una URL IPv4 publica tenemos:

![image](https://github.com/user-attachments/assets/d937caf2-5028-4adf-bf0a-e23e837b68a9)

# Resultado de la prediccion usando EC2

![image](https://github.com/user-attachments/assets/98a4335c-c81f-4314-828c-2c7005c1ff43)


Recordemos que para el problema tratado se clasifica como "0"o "1" la probabilidad de tener un ataque cardíaco.

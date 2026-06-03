import psycopg2
import pandas as pd
from dotenv import load_dotenv
import os

# Charger les variables d'environnement
load_dotenv()

# Connexion à Neon
conn = psycopg2.connect(os.getenv("DATABASE_URL"))

# Lire les données
df = pd.read_sql("SELECT * FROM evenements_inondation", conn)

# Afficher
print(df)
print(f"\nNombre d'événements : {len(df)}")
print(f"\nNiveaux de risque : {df['niveau_risque'].value_counts()}")

conn.close()

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.engine import URL


DATABASE_URL = URL.create(

    drivername='postgresql+psycopg2',

    username='shivam',
    password='&lcq|k9+A3M3)yRX',
    host='backstract-postgres.clcwqmakwzez.ap-south-1.rds.amazonaws.com',
    port=5432,
    database='shivam_test'
)
engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

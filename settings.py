from dotenv import load_dotenv
import os

load_dotenv() #take environment variables from .env

##SERVER INFO
HOST = "0.0.0.0"
PORT = 8000
print("------------------------------")
print("SUMMARY ENVIRONMENT VARIABLES: {}")
print("HOST: {}".format(HOST))
print("PORT: {}".format(PORT))
print("------------------------------")


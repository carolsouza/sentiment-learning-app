#!/usr/bin/env python3
"""
Script simplificado para iniciar o servidor MLflow no Windows
"""
import os
import subprocess
import sys
from pathlib import Path

def start_mlflow_server():
    """Inicia o servidor MLflow de forma simplificada"""

    print(f"Iniciando MLflow server...")
    print(f"Diretório atual: {os.getcwd()}")
    print(f"Server: http://127.0.0.1:5000")
    print("-" * 50)

    # Comando simplificado para MLflow
    cmd = [
        "mlflow",
        "ui",
        "--host", "127.0.0.1",
        "--port", "5000"
    ]

    try:
        # Inicia o servidor
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Erro ao iniciar MLflow: {e}")
        print("Tentando comando alternativo...")

        # Comando alternativo
        cmd_alt = [
            "mlflow",
            "server",
            "--host", "127.0.0.1",
            "--port", "5000"
        ]

        try:
            subprocess.run(cmd_alt, check=True)
        except subprocess.CalledProcessError as e2:
            print(f"Erro também no comando alternativo: {e2}")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\nMLflow server interrompido pelo usuário")
        sys.exit(0)

if __name__ == "__main__":
    start_mlflow_server()